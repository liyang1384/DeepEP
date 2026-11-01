from pathlib import Path
import os
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]

@pytest.mark.parametrize('metadata',[2,4,18])
def test_real_metadata_counts(tmp_path,metadata):
    zig=os.environ.get('DEEPEP_TEST_ZIG')
    if not zig: pytest.skip('Set DEEPEP_TEST_ZIG for CPU validation')
    exe=tmp_path/'metadata-counts.exe'
    subprocess.run([zig,'c++','-std=c++17','-O1','-UNDEBUG',f'-DDEEPEP_LL_METADATA_CORES={metadata}',
        '-I'+str(ROOT/'tests/ll_rdma/metadata_stubs'),'-I'+str(ROOT/'csrc'),
        str(Path(__file__).with_name('metadata_counts_test.cpp')),'-o',str(exe)],check=True,timeout=180)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True,timeout=60)
    assert 'PASS metadata count exchange, prefix, balanced tasks 90 cases' in result.stdout

@pytest.mark.parametrize('profile',[False,True])
@pytest.mark.parametrize('aiv',[16,56,72])
def test_real_receiver_and_stage_wire(tmp_path,profile,aiv):
    zig=os.environ.get('DEEPEP_TEST_ZIG')
    if not zig: pytest.skip('Set DEEPEP_TEST_ZIG for CPU validation')
    exe=tmp_path/'metadata-receiver.exe'
    (tmp_path/'shmem.h').write_text('#pragma once\ninline void dcci_cachelines(uint8_t*,uint64_t) {}\n')
    subprocess.run([zig,'c++','-std=c++17','-O1','-UNDEBUG',f'-DDEEPEP_TEST_AIV={aiv}',
        *(['-DDEEPEP_LL_REFERENCE_PROFILE=1'] if profile else []),'-I'+str(tmp_path),
        '-I'+str(ROOT/'tests/ll_rdma/metadata_stubs'),'-I'+str(ROOT/'csrc'),
        str(Path(__file__).with_name('metadata_receiver_test.cpp')),'-o',str(exe)],check=True,timeout=180)
    result=subprocess.run([str(exe)],check=True,capture_output=True,text=True,timeout=60)
    assert 'PASS stage wire and 42 concurrent receiver reuse cases' in result.stdout
