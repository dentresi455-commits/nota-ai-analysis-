import random
import pytest
from fastapi import HTTPException
from long_alignment import compare_complete


def oracle(a, b):
    prev=list(range(len(b)+1))
    for i,x in enumerate(a,1):
        current=[i]+[0]*len(b)
        for j,y in enumerate(b,1):
            current[j]=min(current[j-1]+1,prev[j]+1,prev[j-1]+(x!=y))
        prev=current
    return prev[-1]


def test_exact_short_errors():
    r=compare_complete([60,62,64,65],[60,63,64,65])
    assert r['edit_distance']==1
    assert r['detailed_feedback']['error_counts']['wrong_pitch']==1
    assert r['all_detected_events_considered']


def test_randomized_against_exact_dp():
    rng=random.Random(1200)
    for _ in range(50):
        a=[rng.randrange(60,66) for _ in range(rng.randrange(2,42))]
        b=[rng.randrange(60,66) for _ in range(rng.randrange(2,42))]
        r=compare_complete(a,b)
        assert r['edit_distance']==oracle(a,b)
        assert sum(r['detailed_feedback']['error_counts'].values())==r['edit_distance']


def test_over_500_notes_not_discarded():
    a=[60,62,64,65,67,69] * 220
    b=a[:]
    b[1175]=71
    r=compare_complete(a,b)
    assert r['reference_count']==1320
    assert r['performed_count']==1320
    assert r['edit_distance']==1
    assert r['detailed_feedback']['errors'][0]['reference_position']==1176


def test_very_long_identical_does_not_allocate_quadratic_matrix():
    seq=[60,62,64,65]*2500
    r=compare_complete(seq,seq.copy())
    assert r['reference_count']==10000 and r['edit_distance']==0


def test_heavy_unrelated_sequences_return_explicit_error(monkeypatch):
    monkeypatch.setenv('NOTA_MAX_ALIGNMENT_CELLS','40000')
    a=[60+i%7 for i in range(310)]
    b=[71+i%7 for i in range(311)]
    with pytest.raises(HTTPException) as x:compare_complete(a,b)
    assert x.value.status_code==422 and 'background' in str(x.value.detail)


def test_empty_rejected():
    with pytest.raises(HTTPException): compare_complete([], [60])


def test_error_sampling_only_not_count_truncated():
    a=[60] * 500
    b=[67] * 500
    r=compare_complete(a,b)
    assert r['edit_distance']==500
    assert len(r['detailed_feedback']['errors'])==100
    assert r['detailed_feedback']['total_error_count']==500
    assert r['detailed_feedback']['errors_sampled']
