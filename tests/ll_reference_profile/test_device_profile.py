"""Audit the reference device recorder placement contract.

These checks do not establish CANN resource usage or device performance.
"""
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[2]
LL = ROOT / 'csrc/kernels/dispatch/low_latency'


def read(relative):
    return (LL / relative).read_text(encoding='utf-8')


def test_reference_points_preserve_business_boundaries_and_async_tail():
    entry = (ROOT / 'csrc/kernels/dispatch/low_latency.cpp').read_text(encoding='utf-8')
    assert entry.index('DeviceProfile::EntryClock') < entry.index('util_set_ffts_config')
    assert entry.index('kernel.Process();') < entry.index('profile.Finish(')
    kernel = read('dispatch_server_dedup_kernel.h')
    for begin, business, end in ((2,'RotateDispatchServerDedupMagic',3), (6,'counts_.SendCnt()',7),
            (8,'sender_.Process(',9), (10,'counts_.RecvCnt()',11), (12,'counts_.CalcCumsum()',13),
            (14,'counts_.CompleteCumsum()',15), (16,'PipeBarrier<PIPE_ALL>()',17),
            (18,'receiver_.Process(',19), (20,'counts_.Finish()',21)):
        assert kernel.index(f'profile_->Mark<{begin}>') < kernel.index(business) < kernel.index(f'profile_->Mark<{end}>')
    send = read('device/dispatch_server_dedup_send.h')
    assert send.index('SendWithoutDedup(input);') < send.index('profile->Mark<39>') < send.index('rdmaBatch_.Flush(')
    assert send.index('rdmaBatch_.Flush(') < send.index('profile->Mark<40>')
    assert 'routes / sendCores_ + (GetBlockIdx() < routes % sendCores_' in send
    counts = read('device/dispatch_server_dedup_counts.h')
    assert counts.index('profile_->Mark<24>') < counts.index('while (sumOfFlag != compareTarget)') < counts.index('profile_->Mark<25>')
    prefix = counts[counts.index('const uint32_t offset = GetCumSum(newAivId);'):]
    assert prefix.index('profile_->Mark<27>') < prefix.index('if (packedFrames_) WaitDispatchClearStatus()')
    recv = read('device/dispatch_server_dedup_recv.h')
    retry = recv.split('do {', 1)[1].split('} while (probe.readyMask == 0U);', 1)[0]
    packet = recv.split('for (uint32_t i = count; i > 0U; --i)', 1)[1].split('++consumed;', 1)[0]
    assert not re.search(r'profile|Clock|Mark|GetSystemCycle', retry)
    assert not re.search(r'profile|Clock|Mark|GetSystemCycle', packet)
    assert recv.count('profile->Clock()') == 5
    assert kernel.index('profile_->Mark<21>') < kernel.index('receiver_.ExportPacketTrace')
    assert '#ifdef DEEPEP_LL_REFERENCE_PROFILE\n        // Three clocks' in recv
    assert recv.index('profile->Mark<35>') < recv.index('profile->Mark<36>') < recv.index('profile->Value<33>(pollCycles)')
    zero_tasks = recv.split('if (tasks_.Count() == 0U)', 1)[1].split('return;', 1)[0]
    for word in (33, 34, 41):
        assert f'profile->Value<{word}>(0U)' in zero_tasks


def test_recorder_has_no_ub_event_or_new_pipeline_barrier():
    device = read('common/ll_reference_profile_device.h')
    assert not re.search(r'\b(?:LocalTensor|TBuf|TQue|TPipe|HardEvent|PipeBarrier|SetFlag|WaitFlag)\b', device)
    assert 'record_[Word] = value;' in device
    assert 'valid_ |= uint64_t(1) << Word;' in device
    assert 'uint64_t values[' not in device
    assert 'kPacketLogWord * sizeof(uint64_t) - 1U' in device
    # No device recorder reads its output buffer to update any duration/count.
    assert not re.search(r'=\s*record_\[', device)
