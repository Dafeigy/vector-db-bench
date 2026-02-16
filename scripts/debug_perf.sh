#!/bin/bash
# debug_perf.sh — 诊断 perf 为什么无法正常工作
set -u

RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[0;33m'
NC='\033[0m'

pass() { echo -e "  ${GREEN}✓${NC} $*"; }
fail() { echo -e "  ${RED}✗${NC} $*"; }
warn() { echo -e "  ${YELLOW}!${NC} $*"; }
info() { echo -e "  $*"; }

echo "=== perf 诊断工具 ==="
echo ""

# 1. 基本环境
echo "[1] 基本环境"
info "内核: $(uname -r)"
info "用户: $(whoami) (uid=$(id -u))"
info "架构: $(uname -m)"
echo ""

# 2. perf 是否安装
echo "[2] perf 安装检查"
if command -v perf &>/dev/null; then
    PERF_PATH=$(command -v perf)
    PERF_VER=$(perf version 2>&1 | head -1)
    pass "perf 已安装: $PERF_PATH"
    info "版本: $PERF_VER"

    # 检查 perf 版本和内核版本是否匹配
    KERNEL_VER=$(uname -r | cut -d- -f1)
    PERF_MAJOR=$(echo "$PERF_VER" | grep -oP '\d+\.\d+' | head -1)
    if [[ "$PERF_MAJOR" == "$KERNEL_VER" ]]; then
        pass "perf 版本与内核匹配 ($PERF_MAJOR)"
    else
        warn "perf 版本 ($PERF_MAJOR) 与内核 ($KERNEL_VER) 可能不匹配"
        info "建议: sudo apt install linux-tools-$(uname -r)"
    fi
else
    fail "perf 未安装"
    info "安装方法:"
    info "  Ubuntu/Debian: sudo apt install linux-tools-$(uname -r)"
    info "  RHEL/CentOS:   sudo yum install perf"
    info "  Arch:          sudo pacman -S perf"
    exit 1
fi
echo ""

# 3. perf_event_paranoid
echo "[3] perf_event_paranoid"
if [[ -f /proc/sys/kernel/perf_event_paranoid ]]; then
    PARANOID=$(cat /proc/sys/kernel/perf_event_paranoid)
    info "当前值: $PARANOID"
    case $PARANOID in
        -1) pass "无限制 (推荐用于调试)" ;;
        0)  pass "允许所有非 raw tracepoint 事件" ;;
        1)  pass "允许用户态 profiling (足够使用)" ;;
        2)  warn "仅允许用户态 profiling，不允许 CPU 事件" ;;
        3)  fail "仅允许 root 使用 perf" ;;
        4)  fail "完全禁用 perf 事件" ;;
        *)  warn "未知值: $PARANOID" ;;
    esac
    if [[ $PARANOID -gt 1 ]] && [[ $(id -u) -ne 0 ]]; then
        info "修复: sudo sysctl kernel.perf_event_paranoid=1"
    fi
else
    fail "/proc/sys/kernel/perf_event_paranoid 不存在"
fi
echo ""

# 4. kptr_restrict
echo "[4] kptr_restrict (内核符号可见性)"
if [[ -f /proc/sys/kernel/kptr_restrict ]]; then
    KPTR=$(cat /proc/sys/kernel/kptr_restrict)
    info "当前值: $KPTR"
    case $KPTR in
        0) pass "内核指针对所有用户可见" ;;
        1) warn "内核指针仅对有 CAP_SYSLOG 的用户可见" ;;
        2) fail "内核指针完全隐藏" ;;
    esac
else
    warn "/proc/sys/kernel/kptr_restrict 不存在"
fi
echo ""

# 5. perf_event_open 系统调用测试
echo "[5] perf 基本功能测试"

# 5a. perf list
echo "  [5a] perf list (可用事件)"
EVENTS=$(perf list 2>&1 | grep -c "event")
if [[ $EVENTS -gt 0 ]]; then
    pass "perf list 返回 $EVENTS 个事件"
else
    fail "perf list 没有返回任何事件"
fi

# 5b. perf stat (最基本的测试)
echo "  [5b] perf stat (基本计数)"
STAT_OUT=$(perf stat -- sleep 0.1 2>&1)
STAT_EXIT=$?
if [[ $STAT_EXIT -eq 0 ]]; then
    pass "perf stat 正常工作"
else
    fail "perf stat 失败 (exit=$STAT_EXIT)"
    info "输出: $(echo "$STAT_OUT" | head -5)"
fi

# 5c. perf record on self (不用 -p)
echo "  [5c] perf record (记录自身进程)"
TMPDIR=$(mktemp -d)
PERF_DATA="$TMPDIR/perf.data"
REC_OUT=$(perf record -o "$PERF_DATA" -F 99 -g -- sleep 0.5 2>&1)
REC_EXIT=$?
if [[ $REC_EXIT -eq 0 ]] && [[ -f "$PERF_DATA" ]]; then
    SIZE=$(stat -c%s "$PERF_DATA" 2>/dev/null || echo 0)
    pass "perf record 成功 (文件大小: ${SIZE} bytes)"
else
    fail "perf record 失败 (exit=$REC_EXIT)"
    info "输出: $REC_OUT"
fi

# 5d. perf record -p PID (附加到外部进程 — 模拟 agent 的用法)
echo "  [5d] perf record -p PID (附加到外部进程)"
sleep 60 &
SLEEP_PID=$!
PERF_DATA2="$TMPDIR/perf2.data"
REC2_OUT=$(timeout 2 perf record -o "$PERF_DATA2" -F 99 -p "$SLEEP_PID" -g 2>&1)
REC2_EXIT=$?
kill $SLEEP_PID 2>/dev/null
wait $SLEEP_PID 2>/dev/null

if [[ -f "$PERF_DATA2" ]]; then
    SIZE2=$(stat -c%s "$PERF_DATA2" 2>/dev/null || echo 0)
    if [[ $SIZE2 -gt 0 ]]; then
        pass "perf record -p PID 成功 (文件大小: ${SIZE2} bytes)"
    else
        fail "perf record -p PID 生成了空文件"
        info "输出: $REC2_OUT"
    fi
else
    fail "perf record -p PID 未生成 perf.data (exit=$REC2_EXIT)"
    info "输出: $REC2_OUT"
fi

# 5e. perf report
echo "  [5e] perf report"
if [[ -f "$PERF_DATA" ]]; then
    RPT_OUT=$(perf report -i "$PERF_DATA" --stdio --no-children 2>&1 | head -20)
    RPT_EXIT=$?
    if [[ $RPT_EXIT -eq 0 ]]; then
        pass "perf report 正常工作"
    else
        fail "perf report 失败 (exit=$RPT_EXIT)"
        info "输出: $(echo "$RPT_OUT" | head -5)"
    fi
else
    warn "跳过 (无 perf.data)"
fi

rm -rf "$TMPDIR"
echo ""

# 6. FlameGraph 工具
echo "[6] FlameGraph 工具链"
if command -v stackcollapse-perf.pl &>/dev/null; then
    pass "stackcollapse-perf.pl 已安装"
else
    warn "stackcollapse-perf.pl 未找到 (flamegraph 生成将跳过)"
    info "安装: git clone https://github.com/brendangregg/FlameGraph && export PATH=\$PATH:\$(pwd)/FlameGraph"
fi
if command -v flamegraph.pl &>/dev/null; then
    pass "flamegraph.pl 已安装"
else
    warn "flamegraph.pl 未找到"
fi
echo ""

# 7. 总结
echo "=== 诊断完成 ==="
