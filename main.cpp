/**
 * ============================================================
 * PTX Assembler Simulator
 * ============================================================
 * Simulates a subset of NVIDIA's PTX (Parallel Thread Execution)
 * ISA. This is a software simulator that:
 *
 * 1. Parses PTX assembly instructions
 * 2. Maintains virtual register file and memory
 * 3. Executes PTX instructions in software
 * 4. Simulates warp execution (32 threads)
 * 5. Reports execution statistics
 *
 * Supported PTX Instructions:
 *   - mov, add, sub, mul, div, rem
 *   - setp (set predicate)
 *   - bra (branch)
 *   - ld, st (load/store)
 *   - mad (multiply-add)
 *   - abs, neg, min, max
 *   - cvt (convert)
 *   - Special registers: %tid, %ntid, %ctaid
 *
 * Author: Jujhaar Singh Aidhen
 * Relevance: PTX ISA, NVIDIA GPU Architecture, PTXAS Compiler
 * ============================================================
 */

#include <iostream>
#include <string>
#include <vector>
#include <map>
#include <unordered_map>
#include <sstream>
#include <stdexcept>
#include <functional>
#include <cassert>
#include <iomanip>
#include <variant>
#include <algorithm>
#include <cstdint>
#include <climits>

// ============================================================
// SECTION 1: PTX TYPE SYSTEM
// ============================================================

enum class PTXType {
    S8, S16, S32, S64,   // Signed integers
    U8, U16, U32, U64,   // Unsigned integers
    F16, F32, F64,       // Floating point
    B8, B16, B32, B64,   // Bit types
    PRED                 // Predicate (boolean)
};

std::string ptxTypeToString(PTXType t) {
    switch(t) {
        case PTXType::S32: return ".s32";
        case PTXType::U32: return ".u32";
        case PTXType::F32: return ".f32";
        case PTXType::F64: return ".f64";
        case PTXType::S64: return ".s64";
        case PTXType::U64: return ".u64";
        case PTXType::PRED: return ".pred";
        case PTXType::B32: return ".b32";
        case PTXType::B64: return ".b64";
        default: return ".u32";
    }
}

PTXType stringToPTXType(const std::string& s) {
    if (s == ".s32" || s == "s32") return PTXType::S32;
    if (s == ".u32" || s == "u32") return PTXType::U32;
    if (s == ".f32" || s == "f32") return PTXType::F32;
    if (s == ".f64" || s == "f64") return PTXType::F64;
    if (s == ".s64" || s == "s64") return PTXType::S64;
    if (s == ".u64" || s == "u64") return PTXType::U64;
    if (s == ".pred" || s == "pred") return PTXType::PRED;
    if (s == ".b32" || s == "b32") return PTXType::B32;
    return PTXType::U32;
}

// ============================================================
// SECTION 2: REGISTER FILE
// ============================================================

struct RegisterValue {
    PTXType type;
    union {
        int32_t  s32;
        uint32_t u32;
        float    f32;
        double   f64;
        int64_t  s64;
        uint64_t u64;
        bool     pred;
    };

    RegisterValue() : type(PTXType::U32), u32(0) {}
    RegisterValue(int32_t v) : type(PTXType::S32), s32(v) {}
    RegisterValue(uint32_t v) : type(PTXType::U32), u32(v) {}
    RegisterValue(float v) : type(PTXType::F32), f32(v) {}
    RegisterValue(double v) : type(PTXType::F64), f64(v) {}
    RegisterValue(bool v) : type(PTXType::PRED), pred(v) {}

    std::string toString() const {
        switch(type) {
            case PTXType::S32: return std::to_string(s32);
            case PTXType::U32: return std::to_string(u32);
            case PTXType::F32: return std::to_string(f32);
            case PTXType::F64: return std::to_string(f64);
            case PTXType::PRED: return pred ? "true" : "false";
            default: return std::to_string(u32);
        }
    }

    int64_t asInt() const {
        switch(type) {
            case PTXType::S32: return s32;
            case PTXType::U32: return (int64_t)u32;
            case PTXType::F32: return (int64_t)f32;
            case PTXType::F64: return (int64_t)f64;
            case PTXType::PRED: return pred ? 1 : 0;
            default: return u32;
        }
    }

    double asFloat() const {
        switch(type) {
            case PTXType::F32: return f32;
            case PTXType::F64: return f64;
            case PTXType::S32: return s32;
            case PTXType::U32: return u32;
            default: return u32;
        }
    }
};

// Register file for one thread
class RegisterFile {
private:
    std::unordered_map<std::string, RegisterValue> regs;

public:
    void set(const std::string& name, RegisterValue val) {
        regs[name] = val;
    }

    RegisterValue get(const std::string& name) const {
        auto it = regs.find(name);
        if (it != regs.end()) return it->second;
        return RegisterValue((uint32_t)0);
    }

    bool exists(const std::string& name) const {
        return regs.count(name) > 0;
    }

    void dump() const {
        std::cout << "  Register File:\n";
        for (auto& [name, val] : regs) {
            std::cout << "    " << std::setw(12) << name
                      << " = " << val.toString() << "\n";
        }
    }
};

// ============================================================
// SECTION 3: PTX INSTRUCTION REPRESENTATION
// ============================================================

struct PTXInstruction {
    std::string opcode;
    PTXType type;
    std::string predicate;   // optional predicate register
    bool negPred = false;    // @!p means negate predicate
    std::vector<std::string> operands;
    std::string label;       // for label definitions
    bool isLabel = false;
    std::string comment;

    std::string toString() const {
        if (isLabel) return label + ":";
        std::string s;
        if (!predicate.empty()) {
            s += "@";
            if (negPred) s += "!";
            s += predicate + " ";
        }
        s += opcode + ptxTypeToString(type);
        for (size_t i = 0; i < operands.size(); i++) {
            s += (i == 0 ? " " : ", ") + operands[i];
        }
        if (!comment.empty()) s += "  // " + comment;
        return s;
    }
};

// ============================================================
// SECTION 4: PTX LEXER/PARSER
// ============================================================

class PTXParser {
private:
    std::string source;
    size_t pos;

    void skipWhitespace() {
        while (pos < source.size() && isspace(source[pos])) pos++;
    }

    std::string readWord() {
        skipWhitespace();
        std::string word;
        while (pos < source.size() &&
               (isalnum(source[pos]) || source[pos] == '_' ||
                source[pos] == '.' || source[pos] == '%' ||
                source[pos] == '-' || source[pos] == '+')) {
            word += source[pos++];
        }
        return word;
    }

    std::string readOperand() {
        skipWhitespace();
        std::string op;
        // Handle [reg+offset] memory addressing
        if (pos < source.size() && source[pos] == '[') {
            while (pos < source.size() && source[pos] != ']')
                op += source[pos++];
            if (pos < source.size()) op += source[pos++]; // ']'
            return op;
        }
        while (pos < source.size() &&
               (isalnum(source[pos]) || source[pos] == '_' ||
                source[pos] == '.' || source[pos] == '%' ||
                source[pos] == '-' || source[pos] == '+' ||
                source[pos] == '[' || source[pos] == ']')) {
            op += source[pos++];
        }
        return op;
    }

    PTXType parseType(const std::string& opcodeWithType) {
        // Extract type suffix like .s32, .f32, etc.
        auto dotPos = opcodeWithType.rfind('.');
        if (dotPos != std::string::npos) {
            return stringToPTXType(opcodeWithType.substr(dotPos));
        }
        return PTXType::U32;
    }

    std::string extractOpcode(const std::string& full) {
        auto dotPos = full.rfind('.');
        if (dotPos != std::string::npos) {
            return full.substr(0, dotPos);
        }
        return full;
    }

public:
    PTXParser(const std::string& src) : source(src), pos(0) {}

    std::vector<PTXInstruction> parse() {
        std::vector<PTXInstruction> instructions;

        std::istringstream stream(source);
        std::string line;

        while (std::getline(stream, line)) {
            // Strip comments
            auto commentPos = line.find("//");
            std::string comment;
            if (commentPos != std::string::npos) {
                comment = line.substr(commentPos + 2);
                line = line.substr(0, commentPos);
            }

            // Trim
            while (!line.empty() && isspace(line.front())) line = line.substr(1);
            while (!line.empty() && isspace(line.back())) line.pop_back();

            if (line.empty()) continue;

            // Skip .version, .target, .address_size, .visible, .entry, etc.
            if (line[0] == '.' && line.find("(") == std::string::npos &&
                line.find("mov") == std::string::npos &&
                line.find("add") == std::string::npos) {
                continue;
            }

            // Label definition
            if (!line.empty() && line.back() == ':') {
                PTXInstruction lbl;
                lbl.isLabel = true;
                lbl.label = line.substr(0, line.size()-1);
                while (!lbl.label.empty() && isspace(lbl.label.back()))
                    lbl.label.pop_back();
                instructions.push_back(lbl);
                continue;
            }

            // Parse instruction
            PTXInstruction instr;
            instr.comment = comment;

            std::istringstream lineStream(line);
            std::string first;
            lineStream >> first;

            // Check for predicate: @%p or @!%p
            if (!first.empty() && first[0] == '@') {
                if (first.size() > 1 && first[1] == '!') {
                    instr.negPred = true;
                    instr.predicate = first.substr(2);
                } else {
                    instr.predicate = first.substr(1);
                }
                lineStream >> first; // read actual opcode
            }

            // Parse opcode.type
            instr.type = parseType(first);
            instr.opcode = extractOpcode(first);

            // Parse operands (comma-separated)
            std::string rest;
            std::getline(lineStream, rest);
            std::istringstream restStream(rest);
            std::string operand;
            while (std::getline(restStream, operand, ',')) {
                while (!operand.empty() && isspace(operand.front())) operand = operand.substr(1);
                while (!operand.empty() && isspace(operand.back())) operand.pop_back();
                // Remove trailing semicolon
                if (!operand.empty() && operand.back() == ';') operand.pop_back();
                if (!operand.empty()) instr.operands.push_back(operand);
            }

            if (!instr.opcode.empty()) {
                instructions.push_back(instr);
            }
        }

        return instructions;
    }
};

// ============================================================
// SECTION 5: PTX VIRTUAL MACHINE / EXECUTOR
// ============================================================

class PTXExecutor {
private:
    RegisterFile regFile;
    std::vector<uint8_t> sharedMem;  // Shared memory (64KB)
    std::vector<uint8_t> globalMem;  // Global memory (simulated)
    std::map<std::string, int> labelMap;
    std::vector<PTXInstruction> instructions;

    // Thread ID (simulating single thread)
    uint32_t threadIdX = 0, threadIdY = 0, threadIdZ = 0;
    uint32_t blockDimX = 32, blockDimY = 1, blockDimZ = 1;
    uint32_t blockIdX = 0, blockIdY = 0, blockIdZ = 0;

    // Execution statistics
    int instrCount = 0;
    int branchCount = 0;
    int memAccessCount = 0;

    RegisterValue resolveOperand(const std::string& op) {
        if (op.empty()) return RegisterValue((uint32_t)0);

        // Special registers
        if (op == "%tid.x") return RegisterValue(threadIdX);
        if (op == "%tid.y") return RegisterValue(threadIdY);
        if (op == "%tid.z") return RegisterValue(threadIdZ);
        if (op == "%ntid.x") return RegisterValue(blockDimX);
        if (op == "%ntid.y") return RegisterValue(blockDimY);
        if (op == "%ctaid.x") return RegisterValue(blockIdX);
        if (op == "%ctaid.y") return RegisterValue(blockIdY);
        if (op == "%laneid") return RegisterValue(threadIdX % 32);
        if (op == "%warpid") return RegisterValue(threadIdX / 32);

        // Immediate value
        if (!op.empty() && (isdigit(op[0]) || op[0] == '-')) {
            try {
                if (op.find('.') != std::string::npos) {
                    return RegisterValue((float)std::stof(op));
                }
                return RegisterValue((int32_t)std::stoi(op));
            } catch (...) {}
        }

        // Hex immediate
        if (op.size() > 2 && op[0] == '0' && op[1] == 'x') {
            return RegisterValue((uint32_t)std::stoul(op, nullptr, 16));
        }

        // Register lookup
        return regFile.get(op);
    }

    bool evaluatePredicate(const std::string& pred, bool negate) {
        RegisterValue val = regFile.get(pred);
        bool result = val.pred;
        return negate ? !result : result;
    }

    void executeInstruction(const PTXInstruction& instr, int& pc) {
        // Check predicate
        if (!instr.predicate.empty()) {
            if (!evaluatePredicate(instr.predicate, instr.negPred)) {
                pc++;
                return; // Predicated out
            }
        }

        instrCount++;
        const auto& ops = instr.operands;

        if (instr.opcode == "mov") {
            if (ops.size() >= 2) {
                regFile.set(ops[0], resolveOperand(ops[1]));
            }
        }
        else if (instr.opcode == "add") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                if (instr.type == PTXType::F32)
                    regFile.set(ops[0], RegisterValue(a.asFloat() + b.asFloat()));
                else
                    regFile.set(ops[0], RegisterValue((int32_t)(a.asInt() + b.asInt())));
            }
        }
        else if (instr.opcode == "sub") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                if (instr.type == PTXType::F32)
                    regFile.set(ops[0], RegisterValue((float)(a.asFloat() - b.asFloat())));
                else
                    regFile.set(ops[0], RegisterValue((int32_t)(a.asInt() - b.asInt())));
            }
        }
        else if (instr.opcode == "mul") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                if (instr.type == PTXType::F32)
                    regFile.set(ops[0], RegisterValue((float)(a.asFloat() * b.asFloat())));
                else
                    regFile.set(ops[0], RegisterValue((int32_t)(a.asInt() * b.asInt())));
            }
        }
        else if (instr.opcode == "div") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                if (instr.type == PTXType::F32) {
                    float bv = b.asFloat();
                    regFile.set(ops[0], RegisterValue(bv != 0 ? (float)(a.asFloat()/bv) : 0.0f));
                } else {
                    int64_t bv = b.asInt();
                    regFile.set(ops[0], RegisterValue((int32_t)(bv != 0 ? a.asInt()/bv : 0)));
                }
            }
        }
        else if (instr.opcode == "rem") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                int64_t bv = b.asInt();
                regFile.set(ops[0], RegisterValue((int32_t)(bv != 0 ? a.asInt()%bv : 0)));
            }
        }
        else if (instr.opcode == "mad") {
            // mad.lo d, a, b, c  =>  d = a*b + c
            if (ops.size() >= 4) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                auto c = resolveOperand(ops[3]);
                if (instr.type == PTXType::F32)
                    regFile.set(ops[0], RegisterValue((float)(a.asFloat()*b.asFloat() + c.asFloat())));
                else
                    regFile.set(ops[0], RegisterValue((int32_t)(a.asInt()*b.asInt() + c.asInt())));
            }
        }
        else if (instr.opcode == "abs") {
            if (ops.size() >= 2) {
                auto a = resolveOperand(ops[1]);
                if (instr.type == PTXType::F32)
                    regFile.set(ops[0], RegisterValue((float)std::abs(a.asFloat())));
                else
                    regFile.set(ops[0], RegisterValue((int32_t)std::abs(a.asInt())));
            }
        }
        else if (instr.opcode == "neg") {
            if (ops.size() >= 2) {
                auto a = resolveOperand(ops[1]);
                if (instr.type == PTXType::F32)
                    regFile.set(ops[0], RegisterValue((float)(-a.asFloat())));
                else
                    regFile.set(ops[0], RegisterValue((int32_t)(-a.asInt())));
            }
        }
        else if (instr.opcode == "min") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                regFile.set(ops[0], RegisterValue((int32_t)std::min(a.asInt(), b.asInt())));
            }
        }
        else if (instr.opcode == "max") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                regFile.set(ops[0], RegisterValue((int32_t)std::max(a.asInt(), b.asInt())));
            }
        }
        else if (instr.opcode == "setp") {
            // setp.cmp.type p, a, b
            // Compare and set predicate
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                bool result = false;

                // The comparison type is embedded in opcode string
                // e.g., setp.lt.s32, setp.eq.u32, setp.ge.f32
                // We'll handle based on what was parsed
                std::string cmpType = instr.opcode;
                if (cmpType.find("lt") != std::string::npos || cmpType == "setp")
                    result = a.asInt() < b.asInt();
                else if (cmpType.find("le") != std::string::npos)
                    result = a.asInt() <= b.asInt();
                else if (cmpType.find("gt") != std::string::npos)
                    result = a.asInt() > b.asInt();
                else if (cmpType.find("ge") != std::string::npos)
                    result = a.asInt() >= b.asInt();
                else if (cmpType.find("eq") != std::string::npos)
                    result = a.asInt() == b.asInt();
                else if (cmpType.find("ne") != std::string::npos)
                    result = a.asInt() != b.asInt();

                regFile.set(ops[0], RegisterValue(result));
            }
        }
        else if (instr.opcode == "selp") {
            // selp.type d, a, b, p  =>  d = p ? a : b
            if (ops.size() >= 4) {
                auto p = regFile.get(ops[3]);
                regFile.set(ops[0], p.pred ? resolveOperand(ops[1]) : resolveOperand(ops[2]));
            }
        }
        else if (instr.opcode == "and") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                if (instr.type == PTXType::PRED)
                    regFile.set(ops[0], RegisterValue(a.pred && b.pred));
                else
                    regFile.set(ops[0], RegisterValue((uint32_t)(a.asInt() & b.asInt())));
            }
        }
        else if (instr.opcode == "or") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                if (instr.type == PTXType::PRED)
                    regFile.set(ops[0], RegisterValue(a.pred || b.pred));
                else
                    regFile.set(ops[0], RegisterValue((uint32_t)(a.asInt() | b.asInt())));
            }
        }
        else if (instr.opcode == "xor") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                regFile.set(ops[0], RegisterValue((uint32_t)(a.asInt() ^ b.asInt())));
            }
        }
        else if (instr.opcode == "shl") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                regFile.set(ops[0], RegisterValue((uint32_t)(a.asInt() << b.asInt())));
            }
        }
        else if (instr.opcode == "shr") {
            if (ops.size() >= 3) {
                auto a = resolveOperand(ops[1]);
                auto b = resolveOperand(ops[2]);
                regFile.set(ops[0], RegisterValue((uint32_t)((uint32_t)a.asInt() >> b.asInt())));
            }
        }
        else if (instr.opcode == "cvt") {
            if (ops.size() >= 2) {
                auto val = resolveOperand(ops[1]);
                // Convert based on dest type
                if (instr.type == PTXType::F32)
                    regFile.set(ops[0], RegisterValue((float)val.asFloat()));
                else if (instr.type == PTXType::S32)
                    regFile.set(ops[0], RegisterValue((int32_t)val.asInt()));
                else
                    regFile.set(ops[0], RegisterValue((uint32_t)val.asInt()));
            }
        }
        else if (instr.opcode == "bra") {
            // Unconditional branch
            if (!ops.empty()) {
                branchCount++;
                auto it = labelMap.find(ops[0]);
                if (it != labelMap.end()) {
                    pc = it->second;
                    return;
                }
            }
        }
        else if (instr.opcode == "ret") {
            pc = (int)instructions.size(); // End execution
            return;
        }
        else if (instr.opcode == "exit") {
            pc = (int)instructions.size();
            return;
        }
        else if (instr.opcode == "ld") {
            // Simplified: ld.global.s32 %rd, [addr]
            memAccessCount++;
            if (ops.size() >= 2) {
                // Just return 0 for simplified simulation
                regFile.set(ops[0], RegisterValue((uint32_t)0));
            }
        }
        else if (instr.opcode == "st") {
            // Simplified store
            memAccessCount++;
        }
        else if (instr.opcode == "bar") {
            // Barrier synchronization - no-op in single thread sim
        }
        else if (instr.opcode == "atom") {
            // Atomic operation - simplified
            if (ops.size() >= 2) {
                regFile.set(ops[0], RegisterValue((uint32_t)0));
            }
        }

        pc++;
    }

public:
    PTXExecutor(uint32_t tidX = 0, uint32_t bidX = 0, uint32_t bdimX = 32)
        : sharedMem(65536, 0), globalMem(1048576, 0),
          threadIdX(tidX), blockIdX(bidX), blockDimX(bdimX) {}

    void buildLabelMap(const std::vector<PTXInstruction>& instrs) {
        instructions = instrs;
        for (int i = 0; i < (int)instrs.size(); i++) {
            if (instrs[i].isLabel) {
                labelMap[instrs[i].label] = i + 1;
            }
        }
    }

    void setRegister(const std::string& name, RegisterValue val) {
        regFile.set(name, val);
    }

    RegisterValue getRegister(const std::string& name) const {
        return regFile.get(name);
    }

    void execute(bool verbose = true) {
        int pc = 0;
        int maxInstr = 100000; // Safety limit
        int executed = 0;

        while (pc < (int)instructions.size() && executed < maxInstr) {
            const auto& instr = instructions[pc];

            if (instr.isLabel) { pc++; continue; }

            if (verbose) {
                std::cout << "  PC=" << std::setw(3) << pc
                          << " " << instr.toString() << "\n";
            }

            int prevPc = pc;
            executeInstruction(instr, pc);
            executed++;
        }
    }

    void dumpState() const {
        std::cout << "\n--- PTX VM State ---\n";
        std::cout << "Thread: (" << threadIdX << "," << threadIdY << ")\n";
        std::cout << "Block:  (" << blockIdX << "," << blockIdY << ")\n";
        regFile.dump();
        std::cout << "\nExecution Stats:\n";
        std::cout << "  Instructions executed: " << instrCount << "\n";
        std::cout << "  Branch count:          " << branchCount << "\n";
        std::cout << "  Memory accesses:       " << memAccessCount << "\n";
    }
};

// ============================================================
// SECTION 6: WARP SIMULATOR (32 threads)
// ============================================================

class WarpSimulator {
private:
    static constexpr int WARP_SIZE = 32;
    std::vector<PTXExecutor> threads;
    std::vector<bool> activeMask;

public:
    WarpSimulator() : activeMask(WARP_SIZE, true) {
        for (int i = 0; i < WARP_SIZE; i++) {
            threads.emplace_back(i, 0, WARP_SIZE);
        }
    }

    void loadProgram(const std::vector<PTXInstruction>& instrs) {
        for (auto& t : threads) {
            t.buildLabelMap(instrs);
        }
    }

    void setRegisterAllThreads(const std::string& reg,
                                std::function<RegisterValue(int)> valueFn) {
        for (int i = 0; i < WARP_SIZE; i++) {
            threads[i].setRegister(reg, valueFn(i));
        }
    }

    void executeAllThreads(bool verbose = false) {
        std::cout << "\n[Warp] Executing " << WARP_SIZE << " threads:\n";
        for (int i = 0; i < WARP_SIZE; i++) {
            if (activeMask[i]) {
                if (verbose) std::cout << "\n  Thread " << i << ":\n";
                threads[i].execute(verbose);
            }
        }
    }

    void dumpResults(const std::string& reg) {
        std::cout << "\n[Warp] Results for register " << reg << ":\n";
        for (int i = 0; i < WARP_SIZE; i++) {
            std::cout << "  Thread[" << std::setw(2) << i << "] = "
                      << threads[i].getRegister(reg).toString() << "\n";
        }
    }
};

// ============================================================
// SECTION 7: SAMPLE PTX PROGRAMS
// ============================================================

// PTX program: Vector addition kernel
std::string vectorAddPTX = R"(
// PTX Vector Addition Kernel
// Each thread computes one element: C[i] = A[i] + B[i]
mov.u32 %r1, %tid.x
mov.u32 %r2, 0
add.s32 %r3, %r1, 10
mul.s32 %r4, %r1, 2
add.s32 %r5, %r3, %r4
)";

// PTX program: Conditional execution with predicates
std::string predicatedPTX = R"(
// Predicated execution example
// Computes: result = (x > 5) ? x*2 : x+1
mov.s32 %r1, 7
mov.s32 %r2, 5
setp.gt.s32 %p1, %r1, %r2
@%p1 mul.s32 %r3, %r1, 2
@!%p1 add.s32 %r3, %r1, 1
)";

// PTX program: Loop with accumulation
std::string loopPTX = R"(
// Loop: sum = 0+1+2+...+(n-1)
mov.s32 %r_sum, 0
mov.s32 %r_i, 0
mov.s32 %r_n, 10
loop_start:
setp.ge.s32 %p_done, %r_i, %r_n
@%p_done bra loop_end
add.s32 %r_sum, %r_sum, %r_i
add.s32 %r_i, %r_i, 1
bra loop_start
loop_end:
)";

// PTX program: FMA (Fused Multiply Add)
std::string fmaPTX = R"(
// FMA: result = a*b + c
mov.f32 %f1, 3
mov.f32 %f2, 4
mov.f32 %f3, 5
mad.f32 %f4, %f1, %f2, %f3
)";

void runPTXProgram(const std::string& name,
                   const std::string& source,
                   bool verbose = true) {
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "PTX PROGRAM: " << name << "\n";
    std::cout << std::string(60, '-') << "\n";
    std::cout << "Source:\n" << source << "\n";
    std::cout << std::string(60, '-') << "\n";

    PTXParser parser(source);
    auto instrs = parser.parse();

    std::cout << "[Parser] Parsed " << instrs.size() << " instructions\n";
    if (verbose) {
        for (auto& i : instrs)
            if (!i.isLabel) std::cout << "  " << i.toString() << "\n";
    }

    PTXExecutor exec(0, 0, 32);
    exec.buildLabelMap(instrs);
    exec.execute(verbose);
    exec.dumpState();
}

int main() {
    std::cout << "PTX ASSEMBLER SIMULATOR\n";
    std::cout << "Simulating NVIDIA PTX ISA execution\n";

    runPTXProgram("Vector Add Thread 0", vectorAddPTX);
    runPTXProgram("Predicated Execution", predicatedPTX);
    runPTXProgram("Loop Accumulation", loopPTX);
    runPTXProgram("FMA Operation", fmaPTX, false);

    // Warp simulation: run all 32 threads
    std::cout << "\n" << std::string(60, '=') << "\n";
    std::cout << "WARP SIMULATION (32 threads)\n";
    std::cout << std::string(60, '-') << "\n";

    PTXParser warpParser(vectorAddPTX);
    auto warpInstrs = warpParser.parse();

    WarpSimulator warp;
    warp.loadProgram(warpInstrs);
    warp.executeAllThreads(false);
    warp.dumpResults("%r5");

    std::cout << "\n[Done] PTX simulation complete!\n";
    return 0;
}
