"""Contracts for count UB lifetimes, publication and prefix reuse."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DEVICE = ROOT / "csrc/kernels/dispatch/low_latency/device"


def _source(name):
    return (DEVICE / name).read_text(encoding="utf-8")


def _block(source, marker):
    """Return the actual braced production block after a unique marker."""
    start = source.index("{", source.index(marker))
    depth = 1
    for end in range(start + 1, len(source)):
        depth += (source[end] == "{") - (source[end] == "}")
        if depth == 0:
            return source[start + 1:end]
    raise AssertionError(f"Unclosed block: {marker}")


def test_count_prefix_is_loaded_only_by_its_output_owner():
    counts = _source("dispatch_server_dedup_counts.h")
    complete = _block(counts, "void ServerDedupCounts::CompleteCumsum()")
    owner = _block(complete, "if (receiver == 0U)")
    before_owner = complete[:complete.index("if (receiver == 0U)")]
    assert "WaitCumSumFlag();" in before_owner  # Every receiver still retires its ready row.
    assert "prefixBuf_" not in before_owner
    assert "DataCopyPad(prefix," not in before_owner
    assert "InitBuffer(prefixBuf_" in owner
    assert owner.index("DataCopyPad(prefix,") < owner.index("SyncFunc<HardEvent::MTE2_S>()")
    assert owner.index("SyncFunc<HardEvent::MTE2_S>()") < owner.index("prefix.GetValue(")
    assert "DataCopyPad(expertTokenNumsOutGMTensor_" in owner
    assert "LocalTensor<int32_t> Prefix()" not in counts
    receiver = _block(_source("dispatch_server_dedup_recv.h"), "void Process(")
    assert receiver.index("pipe_->Reset();") < receiver.index("DataCopyPad(prefix, prefixGlobal")
    assert receiver.index("ServerDedupReceiveSync<HardEvent::MTE2_S>()") < receiver.index("tasks_.Build(prefix")


def test_direct_count_records_live_until_the_existing_reuse_barrier():
    counts = _source("dispatch_server_dedup_counts.h")
    send = _block(counts, "void ServerDedupCounts::SendCnt()")
    assert "routeExchange_" not in counts
    remote = _block(send, "if (remoteOwner)")
    assert remote.index("CountSlots(remoteFrame, first,") < remote.index("LlTransport::DirectCountFrame")
    assert remote.index("LlTransport::DirectCountFrame") < remote.index("SyncFunc<HardEvent::S_V>()")
    assert send.index("LlTransport::DirectCountFrame") < send.index("CountSlots(statusTensor_, start, end - start)")
    assert "Reset(" not in send
    assert "tpipe_->InitBuffer(statusBuf_, ReservedBytes())" in send
    assert "kEpServerDedupCountUbBytes - ReservedBytes()" in send
    assert "return stagePool_;" in _block(counts, "ScratchPool()")
    assert "PipeBarrier<PIPE_ALL>();" in _block(counts, "void Finish()")
    kernel = (DEVICE.parent / "dispatch_server_dedup_kernel.h").read_text(encoding="utf-8")
    assert "sender_.Process(counts_.ScratchPool(), counts_.ReservedBytes(), profile_, counts_.Retirement())" in kernel
    complete = kernel.index("counts_.CompleteCumsum();")
    consume = kernel.index("receiver_.Process(profile_);")
    assert "PipeBarrier<PIPE_ALL>();" in kernel[complete:consume]
    loaded = send.index("DataCopyPad(validExpertIdsTensor_, expertIdsGMTensor_")
    assert "SyncFunc<HardEvent::MTE2_V>();" in send[loaded:send.index("CountSlots(remoteFrame, first,")]


def test_shared_prefix_publication_and_consumers_keep_the_ready_dependency():
    counts = _source("dispatch_server_dedup_counts.h")
    publish = _block(counts, "void ServerDedupCounts::CalRecvAndSetFlag()")
    write = "DataCopyPad(workspaceGlobal[startStatusIndex_], outCountLocal, dataCopyOutParams)"
    assert publish.count(write) == 1
    assert "workspaceGlobal[index" not in publish
    assert publish.index(write) < publish.index("SyncFunc<AscendC::HardEvent::MTE3_V>()")
    assert publish.index("PipeBarrier<PIPE_MTE3>()") < publish.index("layout_.cumsumFlagOffset")
    owner = _block(_block(counts, "void ServerDedupCounts::CompleteCumsum()"), "if (receiver == 0U)")
    assert "DataCopyPad(prefix, workspace, copy, pad);" in owner
    receive = _block(_source("dispatch_server_dedup_recv.h"), "void Process(")
    assert "counts.cntOffset + counts.prefixOffset" in receive
    assert "counts.prefixStride" not in receive
    # Subtotals use one writer-owned slot per receiver, consumers read one contiguous
    # burst, and the slot is reclaimed only after the following shared barrier.
    get_cumsum = _block(counts, "uint32_t ServerDedupCounts::GetCumSum(")
    assert "DataCopyParams sumIntriParams{static_cast<uint16_t>(activeCumSumCores_), 1, 0, 0}" in get_cumsum
    assert "newAivId * UB_ALIGN" not in get_cumsum
    assert "layout_.cumsumOffset + newAivId * UB_ALIGN" in _block(counts, "void ServerDedupCounts::PublishRecvCntSum(")
    wait_flag = _block(counts, "void ServerDedupCounts::WaitCumSumFlag()")
    assert "newAivId * aivUsedCumSum_ * UB_ALIGN" in wait_flag
    assert "if (newAivId < activeCumSumCores_)" in wait_flag
    assert "layout_.cumsumOffset + newAivId * UB_ALIGN" in wait_flag
