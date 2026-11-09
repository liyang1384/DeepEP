from pathlib import Path
ROOT=Path(__file__).resolve().parents[2]

def test_m4_posts_before_m5_and_kernel_owns_retirement():
    root=ROOT/'csrc/kernels/combine/low_latency'
    partial=(root/'combine_server_dedup_partial.h').read_text(encoding='utf-8')
    process=partial.split('uint32_t Process()',1)[1].split('private:',1)[0]
    assert 'rdmaBatch_->SubmitPending();' in process and 'Flush(' not in process
    kernel=(root/'combine_server_dedup_kernel.h').read_text(encoding='utf-8')
    process=kernel.split('void Process()',1)[1].split('private:',1)[0]
    assert process.index('AggregateSourceTokens();') < process.index('PipeBarrier<PIPE_ALL>();') < process.index('returnBatch_.Flush')
    assert 'workspace_, &returnBatch_' in kernel
    assert 'SendReturnPacket(returnBatch_' in kernel and 'SendReturnPacket(*rdmaBatch_' in partial
