import os, math
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
from typing import List, Optional

app = FastAPI(title='NOTA AI analysis service', version='0.1.0')

class CompareRequest(BaseModel):
    reference_midi: List[int] = Field(min_length=1, max_length=1000)
    performed_hz: List[float] = Field(min_length=1, max_length=1000)

@app.get('/health')
def health():
    return {'status': 'ok', 'service': 'nota-ai-analysis', 'version': '0.1.0'}

@app.post('/v1/compare')
def compare(data: CompareRequest, authorization: Optional[str] = Header(None)):
    token = os.getenv('NOTA_API_TOKEN')
    if token and authorization != 'Bearer ' + token:
        raise HTTPException(401, 'Unauthorized')
    if any(not 21 <= x <= 108 for x in data.reference_midi):
        raise HTTPException(422, 'Reference MIDI notes must be 21..108')
    if any(not math.isfinite(x) or not 50 <= x <= 2000 for x in data.performed_hz):
        raise HTTPException(422, 'Invalid performed frequencies')
    performed = [round(69 + 12 * math.log2(x / 440)) for x in data.performed_hz]
    # Edit distance permits skipped/extra notes; no unverified rhythmic scoring.
    a, b = data.reference_midi, performed
    n, m = len(a), len(b)
    dp = [[0]*(m+1) for _ in range(n+1)]
    for i in range(n+1): dp[i][0] = i
    for j in range(m+1): dp[0][j] = j
    for i in range(1,n+1):
        for j in range(1,m+1):
            dp[i][j] = min(dp[i-1][j]+1, dp[i][j-1]+1, dp[i-1][j-1]+(a[i-1]!=b[j-1]))
    edits = dp[n][m]
    score = round(100 * (1 - edits / max(n,m)))
    return {'reference_count': n, 'performed_count': m, 'performed_midi': performed,
            'note_similarity_percent': max(0,score), 'edit_distance': edits,
            'rhythm_scored': False,
            'warning': 'Pitch-sequence estimate only; note extraction and timing must be validated separately.'}
