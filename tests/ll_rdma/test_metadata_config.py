"""Compile-time ownership and build identity checks, without CANN/devices."""
import importlib.util
import os
from pathlib import Path
import subprocess
import pytest

ROOT=Path(__file__).resolve().parents[2]
spec=importlib.util.spec_from_file_location('metadata_build_support',ROOT/'build_support.py')
build=importlib.util.module_from_spec(spec)
spec.loader.exec_module(build)

def test_metadata_environment_validation():
    assert build.metadata_cores({})==4
    for m in range(2,71,2):assert build.metadata_cores({'DEEPEP_LL_METADATA_CORES':str(m)})==m
    for bad in ('','0','1','3','72','-2','4.0',' 4','４',True,4):
        with pytest.raises(ValueError):build.metadata_cores({'DEEPEP_LL_METADATA_CORES':bad})

@pytest.mark.parametrize('metadata',[2,4,6,18,70])
def test_compiled_split_and_balanced_ranges(tmp_path,metadata):
    zig=os.environ.get('DEEPEP_TEST_ZIG')
    if not zig:pytest.skip('Set DEEPEP_TEST_ZIG for CPU checks')
    cpp=tmp_path/'split.cpp'
    cpp.write_text(r'''
#include <cassert>
#include <vector>
#include "kernels/dispatch/low_latency/common/ll_core_config.h"
#include "kernels/dispatch/low_latency/common/ep_memory_server_dedup_count_layout.h"
using namespace DispatchDedup;
int main(){
 assert(kLlCoreProtocolVersion==41000+2*kLlMetadataCores);
 for(uint32_t c=16;c<=72;c+=2){
  assert(MetadataSplitValid(c)==(c>=kLlMetadataCores+2));
  if(!MetadataSplitValid(c))continue;
  uint32_t r=PayloadCores(c);assert(ReceiveBegin(c)==kLlMetadataCores+r && ReceiveBegin(c)+r==c);
  for(uint32_t total:{0U,1U,15U,63U,257U,65536U}){
   uint32_t previous=0,small=total/r,large=(total+r-1)/r;
   for(uint32_t worker=0;worker<r;++worker){
    auto begin=BalancedBegin(total,worker,r),end=BalancedBegin(total,worker+1,r);
    assert(begin==previous && (end-begin==small || end-begin==large));previous=end;
   }
   assert(previous==total);
  }
  auto layout=DispatchServerDedupCountLayout(1024,c);
  assert(layout.outputInfoOffset>=layout.cumsumFlagOffset+kLlMetadataCores*kLlMetadataCores*32);
  assert(layout.outputReadyOffset==layout.outputInfoOffset+r*32);
  assert(layout.counterOffset>=layout.cntOffset+layout.outputReadyOffset+r*32);
  assert(layout.stateBytes<kEpServerDedupControlBytes);
 }
}
''')
    exe=tmp_path/'split.exe'
    command=[zig,'c++','-std=c++17','-O1','-UNDEBUG','-I'+str(ROOT/'csrc'),str(cpp),'-o',str(exe)]
    subprocess.run([*command,f'-DDEEPEP_LL_METADATA_CORES={metadata}'],check=True,timeout=90)
    subprocess.run([str(exe)],check=True,timeout=30)

def test_invalid_macro_is_rejected(tmp_path):
    zig=os.environ.get('DEEPEP_TEST_ZIG')
    if not zig:pytest.skip('Set DEEPEP_TEST_ZIG for CPU checks')
    source=tmp_path/'bad.cpp';source.write_text('#include "ll_metadata_config.h"\n')
    for value in (0,1,3,72):
        result=subprocess.run([zig,'c++','-std=c++17','-fsyntax-only',f'-DDEEPEP_LL_METADATA_CORES={value}',
            '-I'+str(ROOT/'csrc/kernels/dispatch/low_latency/common'),str(source)],capture_output=True,text=True,timeout=60)
        assert result.returncode and 'must be an even integer' in result.stderr
