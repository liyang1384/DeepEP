"""Execute the production owner/index helpers against an independent slot enumeration."""
import ast
from collections import Counter
import math
import os
from pathlib import Path
import random
import subprocess
import pytest

ROOT = Path(__file__).resolve().parents[2]


def test_owner_and_dense_bitmap_index_cover_every_wire_slot(tmp_path):
    zig = os.environ.get('DEEPEP_TEST_ZIG')
    if not zig:
        pytest.skip('Set DEEPEP_TEST_ZIG for CPU validation')
    source = tmp_path / 'owner.cpp'
    source.write_text(r'''
#include <cassert>
#include <vector>
#include <iostream>
#include "kernels/dispatch/low_latency/common/ll_metadata_config.h"
using namespace DispatchDedup;
int main(){
 for(unsigned r=1;r<=36;++r)
 for(unsigned e:{0U,1U,2U,16U,32U,384U,1024U})
 for(unsigned b:{1U,2U,7U,8U,25U,26U,27U,32U,63U,64U,65U,256U}){
  std::vector<unsigned> counts(r);
  for(unsigned region=0;region<e;++region){
   for(unsigned q=0;q<b;++q){
    const unsigned owner=(region+q)%r;
    assert(ForwardOwner(region,q,r)==owner);
    assert(ForwardIndex(region,q,b,owner,r)==counts[owner]++);
   }
   for(unsigned owner=0;owner<r;++owner)
    assert(ForwardSlots(region+1,b,owner,r)==counts[owner]);
  }
  for(unsigned owner=0;owner<r;++owner){
   const unsigned total=ForwardSlots(e,b,owner,r);assert(total==counts[owner]);
   unsigned region=0,q=ForwardFirstQ(0,owner,r);
   for(unsigned j=0;j<total;++j){
    while(q>=b){++region;q=ForwardFirstQ(region,owner,r);}
    assert(region<e && q<b && (region+q)%r==owner);
    assert(ForwardIndex(region,q,b,owner,r)==j);q+=r;
   }
  }
 }
 std::cout<<"PASS all slots, compact bitmap bounds, partial regions, R=1..36\n";
}
''', encoding='utf-8')
    exe = tmp_path / 'owner.exe'
    subprocess.run([zig, 'c++', '-std=c++17', '-O1', '-UNDEBUG', '-I'+str(ROOT/'csrc'),
                    str(source), '-o', str(exe)], check=True, timeout=180)
    result = subprocess.run([str(exe)], check=True, capture_output=True, text=True, timeout=60)
    assert 'PASS all slots' in result.stdout


def test_normal_generator_stays_unchanged_and_uses_both_owner_parities():
    source = ROOT/'tests/elastic/test_ep_low_latency.py'
    tree = ast.parse(source.read_text(encoding='utf-8'))
    names = ('make_random_expert_indices', 'make_routes')
    nodes = [node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in names]
    env = dict(random=random, math=math, BALANCED_ROUTE_SEED=20261014)
    exec(compile(ast.Module(body=nodes, type_ignores=[]), str(source), 'exec'), env)
    routes = env['make_routes'](32, 32, 8, 384, 0, 'normal')
    old, new = [0]*26, [0]*26
    for source_rank, tokens in enumerate(routes):
        counts = Counter()
        for selected in tokens:
            for expert in selected:
                q = counts[expert]; counts[expert] += 1
                if expert//12 == 0:
                    region = source_rank*12+expert%12
                    old[(region*32+q)%26] += 1
                    new[(region+q)%26] += 1
    assert sum(old[1::2]) == 2 and sum(new) == 256
    assert new == [10,11,11,11,10,12,14,13,9,9,11,11,10,8,10,10,7,7,9,7,9,9,10,10,8,10]
