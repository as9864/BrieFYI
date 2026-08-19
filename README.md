### 1. **문제 정의**

뉴스를 통한 새로운 정보가 하루에도 수십건 씩 쏟아지는데, 개인마다 실제로 의미있는 정보를 얻기 위해 매번 찾아 읽고 정리하기에는 어려움

정보 확인의 병목현상 발생

- 관심 분야 업데이트를 일일이 확인하고 검수하는데 시간 소요
- 정보 과다로 새 정보에서 중요한 정보만 걸러내기 어려움
- 여러 채널을 오가며 정보 체크하기 어려움

BrieFYI는 이러한 문제에서 출발했습니다. 

관심 키워드로 뉴스를 매일 자동 수집하고 LLM으로 핵심 인사이트를 뽑아 이메일로 정리해 보내주면, 사람이 직접 뉴스를 확인하는 시간을 줄일 수 있다고 생각했습니다.

### 2. 문제 해결 방법

## **2.1 핵심 알고리즘**

### **2.1.1 검색 — Hybrid Search + RRF**

- 임베딩 모델: BGE-M3
    - 다국어 임베딩으로 범용성과 성능이 적당히 높고, 한글/영어에 맞춰 fine tuning된 후속 모델들도 있어 Baseline으로 사용하기에 적절하다고 판단.

원문 텍스트 랭킹(`ts_rank_cd`)과 임베딩 코사인 유사도를 **RRF**(Reciprocal Rank Fusion, vector 7 : text 3, k=60)로 합산. k=60은 RRF 원 논문의 경험적 최적값으로, 고순위 결과의 점수 쏠림을 정규화하는 역할.

### **2.1.2 4-Layer Indexing — GLiNER2**

기사에서 **대분류/도메인/엔티티/사건**을 뽑아 계층화된 메타데이터를 만들고, 검색 시 질의와 일치하면 가산점을 주는 구조. 분류·개체·관계 추출을 스키마 하나로 처리하는 경량 모델(GLiNER2)을 채택 — LLM API 호출보다 비용·지연이 낮다. 원문 76건 전량 인덱싱 성공(0건 실패, chunk 355개·event 327개).

**품질 평가(신규, n=20 표본)**: 인덱싱된 76건 중 20건을 무작위 추출해, Claude가 원문 제목·요약만 보고(GLiNER2는 본문 전체를 봤으므로 이 평가는 정보가 더 적은 상태에서 하는 보수적 채점) category/entity/event를 직접 채점.

| 항목 | 결과 |
| --- | --- |
| 카테고리 정확도 | 16/20 (80%) — 광고성 기사·행사 공지·스포츠 기사가 "기술"/"경제"로 잘못 분류된 사례 4건 |
| 도메인 태깅 | 노이즈 큼 — 산불 기사에 "부동산·로봇", AI 게재금지 사설에 "로봇" 등 무관한 태그 다수 |
| 엔티티 | 대체로 타당하나 일반명사를 엔티티로 오인(`Students`, `cantautore`)하거나 본문과 무관한 브랜드명이 섞인 사례 존재 |
| **이벤트/관계** | **표본 20건 중 약 10건에서 명백히 잘못된 관계 발견** — "BTS 멤버 V가 Bayern Munich에 임명, BTS 탈퇴"(사실무근), "DeepSeek가 OpenAI·Huawei를 인수", "Trump가 US에서 사임" 등. 한 기사에서 46개 이벤트(평균 5.3건의 8.7배)가 생성되며 중복·저품질 관계가 다수 포함 |

**결론**: 카테고리 분류는 실사용 가능한 수준(80%)이지만, **이벤트/관계 추출은 환각 비율이 높아 후처리 없이 바로 쓰기 어렵다** — 다음 단계에서 신뢰도 필터링이나 검증 단계가 필요. (표본이 작고 Claude 자체 판정이라 완전한 정답 라벨은 아님 — 방향성 확인 수준)

### **2.1.3 리랭킹 — Cross-Encoder**

BGE-M3 벡터 검색 후보를 BGE-reranker-v2-m3로 재정렬. 벡터 유사도만으로는 실질 관련성 구분에 한계가 있어 도입.

- 1차: BEG-M3 임베딩+pgvector로 질문과 유사한 문서 검색
벡터 유사도만으로 실사용자의 질문과 관련이 높은 문서를 구분하는 데 한계→리랭킹 적용
    - 사용모델: beg-reranker-v2-m3
    - 벡터 검색에서 가져온 후보 문서들을 다시 질문과 비교
    → 각 문서 리랭커 점수 계산
    → 질문과 실제 내용이 연관이 높으면 순위를 높임

### **2.1.4 클러스터링 — 인사이트 학습 데이터 생성 (`make_train_data/`)**

기사 간 연관관계를 4단계로 판단하는 알고리즘을 직접 설계:

1. **시간창 필터** — narrow(72h, 같은 사건) / broad(28d, 같은 테마) 두 창을 병행 운영
2. **엔티티 자카드 후보군** — union-find로 공통 엔티티 비율이 높은 기사끼리 묶음
3. **임베딩 그리디 정제** — 후보군 내에서 코사인 유사도 임계치로 세부 클러스터 분리
4. **시간창 제약 근접중복 병합** — 임베딩만으로 먼 시간대 기사가 합쳐지는 것을 방지(1단계 필터가 4단계에서 무력화되지 않도록)

### **2.1.5 단발성 사실 탐지 — 환각 방지 보정**

클러스터에 들지 못한 기사 중 일부를 의도적으로 "특별한 함의가 없는 단발성 사실"로 표시해 전체 학습 데이터의 **15~20%**를 채운다. 모델이 모든 기사에서 억지로 인사이트를 만들어내는 환각을 방지하기 위한 calibration 전략.

### **2.1.6 요약 검증 — LLM-as-Judge 재시도 루프 (`summarize_agent`)**

검색 결과 → **요약 생성 → grounding 검증(근거 대조) → LLM judge 채점** → 점수 미달 시 재생성, 최대 시도 횟수까지 반복. 시도마다 DB에 이력을 남겨 재현·분석 가능하게 설계.

### **2.1.7 LLM-as-Judge 채점 + 프롬프트 개선 3라운드 루프**

가중치는 고정한 채, 추론 시 system prompt만 바꿔가며 골든 held-out셋에 생성 → **새로 설계한 LLM-as-judge**(Claude Haiku, `classification_match`/`grounded`/`insight_quality` 3축 채점)로 평가하는 루프 구성

| 라운드 | 프롬프트 특징 | classification_match | grounded_rate | quality |
| --- | --- | --- | --- | --- |
| R1 (베이스라인) | 현행 프롬프트 그대로 | 0.909 | 0.636 | 51.7 |
| R2 | "애매해도 시도하라" 추가 | 0.923 | 0.462 ↓ | 35.0 ↓ |
| R3 | "적극성" 빼고 정확성 규칙만 강화 | 0.533 ↓↓ | 0.600 | 35.0 |

**세 지표를 동시에 개선한 라운드는 없었다** — 오히려 R3의 "애매하면 안전하게 no_strong_insight를 선택하라"는 지시가, **v2 학습 데이터 불균형이 만들었던 것과 똑같은 클래스 붕괴를 프롬프트 레벨에서 재현**했다(insight 8건 중 7건을 놓침). 데이터든 프롬프트든, 이 모델은 같은 지름길로 수렴하는 경향이 있다는 것을 두 경로로 확인한 셈이다.

**표본을 15건→held-out 전체 28건으로 늘려 베이스라인(R1)을 최종 검증**한 결과:

| 항목 | 값 |
| --- | --- |
| classification_match_rate | **85.7%** (24/28) |
| grounded_rate | 57.1% |
| 혼동행렬 | TP 9 / TN 15 / FP 2 / FN 2 |

n=15에서 좋아 보였던 90.9%가 n=28에서 85.7%로 소폭 내려온 것은 작은 표본의 우연이 큰 표본에서 실제 수준으로 수렴한 것으로 보이며, **FP(2)와 FN(2)이 처음으로 균형을 이룬 것**이 가장 중요한 신호다 — v2 직후의 "무조건 다수 클래스" 붕괴나 R3의 "안전 편향" 붕괴와 달리, 지금은 양방향으로 비슷하게 틀린다.

## **2.2 파인튜닝 — 용량 대비 추천 모델**

![alt text](image.png)

요약·인사이트 추출은 추론(reasoning)보다 지시 이행·문체 일관성이 중요한 태스크라, 파라미터를 무작정 키우기보다 **8B 안팎 구간을 잘 파인튜닝하는 쪽이 비용 대비 효율이 높다**고 판단함. 

![alt text](image-1.png)

모델 종류 선정에는 위 생태계 지형도(한국어 특화 / 글로벌 다국어 / 멀티모달 / 주의·비추천 4분면)를 참고함

학습 실행은 **EXAONE-3.5-7.8B-Instruct**로 진행했다 — 한/영 균형에 최적화돼 있고 현재 상황에 가능한 크기대이기 때문. 

## **2.3 데이터 처리**

- **누수 방지**: 학습/검증/테스트 데이터 간 중복 및 유사 문장을 완전히 제거(0건)하여 데이터 누수를 차단.
- **Completion-only 마스킹**: 프롬프트 구간은 loss 계산에서 제외(`100` 마스킹)하고, 응답(요약문) 구간에만 loss를 부여해 모델이 원문을 그대로 복사하는 현상을 방지.
- **자동 모듈 감지**: EXAONE 특유의 레이어 이름(`out_proj`, `c_fc_0` 등)을 자동 스캔해 LoRA 어댑터를 올바르게 부착.
- **검색 단위 세분화**: 뉴스 원문을 그대로 검색하면 정확도가 떨어질 수 있어 500 토큰(BGE-M3 tokenizer 기준) 단위, 50 토큰 오버랩으로 chunking.
- 각 chunk를 BGE-M3로 벡터화해 pgvector에 저장.
- **검색 범위 세분화**: 기사별 대분류/도메인/엔티티/이벤트 metadata를 생성(위 4-Layer indexing).

## **2.4 학습 전략 (데이터 구성 · 학습 방법 · 평가 방법)**

### **학습 데이터**

**구성**: AI Hub 「문서요약 텍스트」(신문기사·사설·법률 3종) + HuggingFace `daekeun-ml/naver-news-summarization-ko`를 출처가 균형을 이루도록 혼합(총 약 2만 건).

| Split | 총계 | aihub_editorial | aihub_law | aihub_news | naver_news |
| --- | --- | --- | --- | --- | --- |
| train | 19,178 | 4,997 | 4,896 | 4,985 | 4,300 |
| validation | 998 | 249 | 250 | 249 | 250 |
| test | 1,977 | 497 | 498 | 499 | 483 |

### **학습 방법**

- **기법**: QLoRA(4-bit NF4 양자화 + compute dtype bf16), 학습 파라미터 비율 약 0.53%
- **하이퍼파라미터**: AdamW 옵티마이저, Sequence Length 1,536 tokens, Effective Batch Size 16, Cosine 스케줄러로 2 에폭

| 항목 | 값 |
| --- | --- |
| Base model | EXAONE-3.5-7.8B-Instruct (7.86B params) |
| Adaptation | QLoRA — 4-bit NF4 + double quant, compute dtype bf16 |
| LoRA | r=16, α=32, dropout 0.05, bias none |
| Target modules | `q_proj`, `k_proj`, `v_proj`, `out_proj`, `c_fc_0`, `c_fc_1`, `c_proj` (자동 감지) |
| Trainable params | **41.9M / 7.86B (0.53%)** |
| Objective | completion-only CE — 프롬프트 토큰은 `-100`으로 마스킹 |
| Seq. length | 1,536 tokens (target ≤ 256) |
| Optimizer | `adamw_torch_fused`, lr 1e-4, cosine, warmup 3%, wd 0, clip 1.0 |
| Batch | 4 × grad-accum 4 = **effective 16**, `group_by_length`, grad checkpointing, SDPA |
| Schedule | **1 epoch = 1,199 steps** |
| Hardware | 1× RTX 5070 Ti 16GB (sm_120) — **11.1 h** (39,795 s, 0.48 samples/s) |
| Seeds | 1 run, seed 42 (분산 미측정) |

### **평가 방법 (요약)**

- **지표:** 한국어 어절(`word`) 기준 **ROUGE-1 / ROUGE-2 / ROUGE-L** 측정.
- **비교 기준:** 본문 앞 3문장을 그대로 가져오는 **Lead-3 베이스라인**과 비교 평가.

### 3. 프로젝트 소개

## **3-1. 시스템 간략 설명**

**BrieFYI**는 관심 키워드로 뉴스를 매일 자동 수집해 LLM으로 핵심 요약·인사이트를 뽑아 이메일로 보내주는 뉴스 다이제스트 시스템이다.

핵심 파이프라인(뉴스 수집 → 요약/인사이트 → 이메일 발송) 위에, 이를 뒷받침하는 세 개의 독립 서브시스템이 같은 PostgreSQL(+pgvector) DB를 공유하며 느슨하게 결합돼 있다.

| 구분 | 역할 |
| --- | --- |
| **메인 파이프라인** | 매일 실행되는 다이제스트 본체. 수집 → 저장 → 요약 → 인사이트 → 이메일 |
| **데이터 파이프라인** | 파인튜닝용 학습 데이터 볼륨을 늘리기 위해 원문 기사를 LLM으로 태깅·정규화 |
| **RAG 저장·검색 파이프라인** | 원문을 벡터 DB에 인덱싱하고, 검색 결과를 요약·검증·재시도하는 에이전트 제공 |
| **파인튜닝 파이프라인** | 자체 요약 모델 학습(LoRA/QLoRA) + 인사이트 생성용 학습 데이터 자동 준비 |

## **3-2. 시스템 구조도 (Blueprint)**

### **전체 구조도**

```mermaid
flowchart TB
    subgraph Trigger["트리거"]
        CRON["GitHub Actions cron\n(daily-digest.yml, 매일 08:00 KST)"]
        SCHED["trigger/scheduler.py\n(main.py --mode trigger)"]
    end

    subgraph Main["main.py"]
        INITDB["init_db()"]
        DIGEST["run_digest()"]
    end

    subgraph Graph["graph/pipeline.py (LangGraph, 고정 순서)"]
        direction LR
        N1["fetch_news"] --> N2["store_raw"] --> N3["summarize"] --> N4["extract_insight"] --> N5["format_email"] --> N6["send_email"]
    end

    subgraph Agents["agents/"]
        COLLECTOR["CollectorAgent"]
        SUMMARIZER["SummarizerAgent"]
        DISTRIBUTOR["DistributorAgent"]
    end

    subgraph ExtAPI["외부 API"]
        GNEWS[("GNews API")]
        CLAUDE[("Anthropic API")]
        RESEND[("Resend API")]
    end

    subgraph DB["PostgreSQL 16 + pgvector"]
        RAWART[("raw_articles")]
        DIGESTS[("digests")]
        SENDLOG[("send_log")]
        ENRICH[("enrichment / synonym_table")]
        VEC[("article_chunks / chunk_embeddings\narticle_topics / article_events\nsummarize_agent_runs")]
    end

    subgraph DataGen["data_pipeline/ (독립 Docker, Groq LLM)"]
        DP["extract → enrich → normalize"]
    end

    subgraph FT["finetune/ (독립 패키지)"]
        FTP["digests_export / enrichment_export\n→ LoRA/QLoRA 학습"]
        MTD["make_train_data/\n클러스터링 → 단발성 탐지 → export"]
    end

    subgraph RAGL["rag_latest/ (독립 CLI)"]
        RAGCLI["content → embed → extract\n→ retriever(hybrid+rerank)"]
        AGENT["summarize_agent\n요약 → 검증 → judge → 재시도"]
    end

    CRON --> DIGEST
    SCHED --> DIGEST
    DIGEST --> INITDB --> RAWART & DIGESTS & SENDLOG & ENRICH & VEC
    DIGEST --> Graph
    N1 -->|위임| COLLECTOR --> GNEWS
    N2 --> RAWART
    N3 -->|위임| SUMMARIZER --> CLAUDE
    N4 -->|위임| SUMMARIZER
    SUMMARIZER --> DIGESTS
    N6 -->|위임| DISTRIBUTOR --> RESEND
    DISTRIBUTOR --> SENDLOG

    RAWART -->|읽기| DP --> ENRICH
    DIGESTS -->|export| FTP
    ENRICH -->|export| FTP
    ENRICH -->|기사 조회| MTD

    RAWART -->|읽기| RAGCLI --> VEC
    VEC -->|검색 결과| AGENT --> VEC
```

### **간략 흐름도 (4대 파이프라인)**

```mermaid
flowchart LR
    subgraph 메인["메인 파이프라인"]
        direction LR
        A1["뉴스 수집"] --> A2["DB 저장"] --> A3["요약 · 인사이트 생성"] --> A4["이메일 발송"]
    end
```

```mermaid
flowchart LR
    subgraph 데이터["데이터 파이프라인"]
        direction LR
        B1["뉴스 수집"] --> B2["키워드 추출"] --> B3["LLM 분류 · 태깅"] --> B4["용어 정규화"]
    end
```

```mermaid
flowchart LR
    subgraph RAG["RAG 저장 파이프라인"]
        direction LR
        C1["기사 본문 수집"] --> C2["문단 분할"] --> C3["임베딩 생성"] --> C4["벡터 DB 저장"]
    end
```

```mermaid
flowchart LR
    subgraph 배포["배포 워크플로우"]
        direction LR
        D1["수신자 목록"] --> D2["개별 발송"] --> D3["성공 · 실패 기록"] --> D4["결과 요약"]
    end
```

---

## **3-3. 시스템 모듈별 간략 작업 흐름 및 설명**

### **① 메인 다이제스트 파이프라인 — `main.py` / `graph/pipeline.py` / `agents/`**

```mermaid
flowchart LR
    S(("시작")) --> N1["fetch_news"] -->|GNews API| N2["store_raw"] --> N3["summarize"] -->|Anthropic API| N4["extract_insight"] --> N5["format_email"] --> N6["send_email"] -->|Resend API| E(("종료"))
```

- 설명: LangGraph `StateGraph`로 조건 분기 없이 고정 순서 실행. 노드는 실제 로직을 갖지 않고 `agents/registry.py`가 조립한 에이전트(Collector/Summarizer/Distributor)에 위임하는 얇은 어댑터다.
- **핵심 포인트**: 노드 실패는 `PipelineState.error`에 기록되고 `main.py`가 이를 보고 종료 코드를 결정한다. `DistributorAgent`는 다중 수신자를 개별 발송·개별 로깅하며, **전원** 실패했을 때만 상위로 실패를 알려 CI가 "메일 0통인데 초록불" 상황을 놓치지 않는다.

### **② 데이터 파이프라인 — `data_pipeline/`**

```mermaid
flowchart LR
    A["뉴스 수집\n(다중 키워드)"] --> B["키워드 추출\n(KeyBERT, 로컬)"] --> C["LLM 분류·태깅\n(Groq, insights/category/domain/entity/event)"] --> D["용어 정규화\n(exact→fuzzy→LLM fallback)"]
```

- 설명: 별도 Docker 이미지로 완전히 독립 배포되는 서브시스템. `raw_articles`를 읽어 무료 티어 Groq LLM으로 태깅한 뒤, `enrichment`/`synonym_table`에 정규화된 값을 쓴다. 목적은 파인튜닝용 학습 데이터 볼륨 확보.
- **핵심 포인트**: `pipeline_status` 컬럼으로 각 기사가 어디까지 처리됐는지 추적하므로, 무료 API 환경에서 배치가 중간에 끊겨도 다음 실행이 이어받는다(재시작 안전).

### **③ RAG 저장·검색 파이프라인 — `rag_latest/`**

```mermaid
flowchart LR
    A["기사 본문 수집\n(실패 시 title+description 대체)"] --> B["청킹\n(BGE-M3 tokenizer, 500토큰/50오버랩)"] --> C["임베딩 생성\n(HF Inference API)"] --> D["벡터 DB 저장\n(chunks/embeddings/topics/events)"]
    D --> R["하이브리드 검색\n(vector+text RRF, cross-encoder rerank)"]
    R --> AG["요약 에이전트\n요약→grounding검증→LLM judge→재시도"]
```

- 설명: 원문을 벡터 DB에 인덱싱(청킹+임베딩+GLiNER2 4-Layer 메타데이터/구조화 이벤트)하고, 하이브리드 검색 결과를 받아 요약을 생성·검증·재시도하는 `summarize_agent`를 제공한다. 요약이 근거(grounding) 검증을 통과하고 LLM judge 점수가 임계치 이상일 때까지 재생성하며, 시도마다 `summarize_agent_runs`에 기록을 남긴다.
- **핵심 포인트**: 본문 수집이 막혀도(사이트 차단 등) `title+description`으로 대체해 전체 배치를 막지 않는다. `graph/pipeline.py`는 아직 이 패키지를 import하지 않는 독립 CLI 단계.

### **④ 파인튜닝 파이프라인 — `finetune/`**

```mermaid
flowchart LR
    subgraph 학습["요약 모델 학습"]
        direction LR
        F1["digests / enrichment export"] --> F2["JSONL 학습 데이터"] --> F3["Colab에서 LoRA/QLoRA 학습"]
    end
    subgraph 인사이트["인사이트 학습 데이터 생성 (make_train_data/)"]
        direction LR
        M1["DB에서 기사 조회"] --> M2["엔티티 추출"] --> M3["클러스터링\n(시간창+엔티티+임베딩)"] --> M4["단발성 후보 선정\n(15~20% 쿼터)"] --> M5["클러스터 단위 파일 export\n(Claude 채팅용 프롬프트 포함)"]
    end
```

- **무엇을 하나**: 두 갈래로 구성된다. 
(1) `digests_export.py`/`enrichment_export.py`로 라이브 파이프라인 산출물을 JSONL로 뽑아 Colab에서 LoRA/QLoRA로 요약 모델을 학습한다. 
(2) `make_train_data/`는 DB에 쌓인 원문을 시간창(72h/28d)+엔티티 자카드+임베딩 유사도로 클러스터링해 "연관 기사 묶음"과 "단발성 사실 기사"를 분리하고, 클러스터 단위 파일로 export한다 — 사람이 이 파일을 Claude 채팅에 붙여넣어 정답(facts/insights/테스트 Q&A)을 생성하는 수동 골드 생성 워크플로우로 이어진다.
- **오늘 기준 실행 결과**: 인덱싱된 76건 기준 클러스터 4개(narrow 2 + broad 2) + 단발성 배치 1개(13건)를 `finetune/make_train_data/output/`에 생성.
- **핵심 포인트**: 환각 방지를 위해 전체의 15~20%를 "특별한 함의가 없는 단발성 사실" 기사로 의도적으로 채우는 보정(calibration) 전략을 씀. 요약 모델 학습 자체는 스캐폴드 완성 단계이며 실제 학습 실행 이력은 아직 없다.

### **⑤ DB 계층 — PostgreSQL 16 + pgvector**

- **무엇을 하나**: `db/db.py`의 `init_db()`가 앱 시작 시마다 `db/schema.sql`(운영 테이블 + data_pipeline용 `enrichment`/`synonym_table`)과 `db/vector_schema.sql`(RAG용 벡터 테이블 + `summarize_agent_runs`)을 idempotent하게 적용한다.
- **핵심 포인트**: 위 네 파이프라인이 실제로는 서로 직접 호출하지 않고, 이 DB 하나를 공유 저장소 삼아 느슨하게 결합돼 있다 — 각자 독립 배포·독립 실행이 가능한 이유.

## **3-4. 시연**

<video controls src="docs/브리피 시연 영상_260819.mp4" title="Title"></video>

### 4. 평가 방법 (모두)

<aside>
💡

수행하였던 방법론을 어떻게 평가하였는지 작성되어야 합니다

- 양적평가(Quantitatively)의 경우 어떤 분석을 수행하고 어떤 평가 지표를 사용하였는지
- 질적평가(Qualitatively)의 경우 어떤 관점에서 평가하였고 어떤 시사점을 보여주는지
- 제안하는 지표들을 어떻게 다른 연구 결과들과 비교하여 우수성을 보여줄 수 있는지
    
    or 기존에 비해 어떤 차별점을 가지며 개선된 시스템을 선보였는지
    
</aside>

## **RAG 평가 — 검색 성능**

**목적**: 리랭커(BGE-reranker-v2-m3) 도입이 실제로 검색 품질을 높이는지 확인.

- **모델**: BGE-M3(임베딩) + BGE-reranker-v2-m3(리랭커)
- **평가 지표**: Recall@k, MRR
- **정답 지정 방식**: DB에 있는 612개 SBS 기사 중 `article_id`(고유 번호)로 매칭
- **쿼리**: 4개 (야구/삼성/축구/반도체)

!image.png

**시행착오: 정답 매칭 방식에 따라 결론이 완전히 달라졌다**

**1차 — 기사 제목으로 매칭**했을 때는 리랭커 도입 후 오히려 지표가 나빠지는 것처럼 보였다:

!image.png

- Recall@1: 0.8571 → 0.7143 (**0.1429**)
- Recall@3: 0.8571 → 1.0000 (+0.1429)
- Recall@5: 1.0000 → 1.0000 (변화 없음)
- MRR: 0.8929 → 0.8571 (**0.0357**)

제목 문자열 매칭은 같은 사건을 다루는 유사 제목 기사(예: 같은 리드로 재작성된 여러 매체 기사)를 서로 다른 "정답"으로 잘못 셀 수 있어, 정답 판정 자체의 노이즈가 리랭커 효과를 가리고 있었다.

**2차 — `article_id`(고유 번호)로 매칭**하도록 정답 판정 방식을 바꾸자:

!image.png

전체 4개 쿼리 평균 MRR 1.0000 → 1.0000, Recall@10 1.0000(전/후 동일), Precision@10 0.1000(전/후 동일) — **벡터 검색 단계에서 이미 만점**이라 리랭커 적용 전/후 차이가 전체 평균에서는 보이지 않았다.

### **리랭커 효과가 없던 건 아님 — 개별 쿼리 단위 확인**

전체 평균이 이미 천장(ceiling)에 붙어 있어 집계 지표로는 안 보였을 뿐, "삼성전자" 쿼리 하나만 떼어 보면 순위 개선이 뚜렷했다.

**리랭커 적용 전** — pgvector 코사인 거리 기준 정답("삼성전자, 3분기 반도체 부문 실적 발표")이 4위:

!image.png

**리랭커 적용 후** — BGE-reranker-v2-m3 점수로 재정렬하면 정답이 2위로 올라온다:

!image.png

**정리**:

!image.png

- 정답 순위: 4위 → 2위
- Recall@1: 0 → 0 (변화 없음)
- Recall@3: 0 → 1
- Recall@5: 1 → 1 (변화 없음)
- MRR: 0.2500 → 0.5000

**결론**: 4개 쿼리 전체 평균으로만 보면 리랭커 효과가 "없다"는 착시가 생기지만, 이는 평가셋이 작고(4개 쿼리) 이미 벡터 검색만으로 천장에 가까웠기 때문이다. 
개별 쿼리 단위로는 리랭커가 실제로 순위를 끌어올리는 사례가 확인됐다. 다만 표본이 4개 쿼리·612개 기사로 작아, 이 결과를 "리랭커가 일반적으로 몇 % 개선한다"는 식으로 일반화하기는 이르다 — 쿼리 수를 늘린 재평가가 필요한 지점으로 남겨둔다.

## **파인튜닝 평가 — 요약 품질**

테스트 세트 1,977건 중 **앞 200건**, ROUGE-1/2/L F1(%), `word` 분절, batch 4 greedy 기준. 네 시스템을 같은 문서·같은 생성 설정으로 비교했다.

| 시스템 | 내용 |
| --- | --- |
| **lead-3** | 본문 앞 세 문장 복사 — 학습이 필요 없는 하한선 |
| **base-0shot** | 파인튜닝 전 베이스 모델. 어댑터와 **완전히 같은** 프롬프트·생성 설정 |
| **base-prompted** | 파인튜닝 전 베이스 모델. 프롬프트로 "2~3문장·150자 내외·요약문만" 지시 |
| **qlora** | 학습한 어댑터 |

**ROUGE-1 F1 (%)** — 각 행의 최고값을 굵게 표시.

| 데이터 | n | lead-3 | base-0shot | base-prompted | qlora |
| --- | --- | --- | --- | --- | --- |
| 전체 | 200 | 38.86 | 18.72 | 19.89 | **39.09** |
| aihub_editorial | 49 | 19.61 | 11.49 | 14.06 | **24.89** |
| aihub_law | 47 | **44.05** | 18.04 | 17.17 | 42.39 |
| aihub_news | 53 | **42.17** | 19.32 | 21.51 | 35.46 |
| naver_news | 51 | 49.15 | 25.65 | 26.31 | **53.48** |

전체 200건 기준 **R-2 / R-L**: lead-3 `29.50 / 34.04` · base-0shot `7.32 / 14.64` · base-prompted `7.48 / 15.79` · qlora `28.01 / 35.58`.

**ΔR-1과 95% 신뢰구간** — 문서 단위 paired bootstrap(10,000회, seed 0). CI가 0을 포함하면 그 차이는 표본 오차와 구별되지 않는다. 

(표의 수치 값은 모두 ROUGE-1 F1 점수를 의미함)

(수치 값의 증감 기준은 qlora를 기준으로 각 모델 대비 차이를 의미함)

| 데이터 | qlora vs base-0shot | 95% CI | qlora vs base-prompted | 95% CI | qlora vs lead-3 | 95% **CI** |
| --- | --- | --- | --- | --- | --- | --- |
| 전체 | **+20.38** | [+18.07, +22.75] | **+19.20** | [+16.85, +21.66] | +0.23 | [−2.21, +2.74] |
| aihub_editorial | **+13.40** | [+9.48, +17.52] | **+10.83** | [+7.02, +14.78] | **+5.28** | [+1.14, +9.36] |
| aihub_law | **+24.35** | [+19.19, +29.55] | **+25.21** | [+19.61, +30.81] | −1.66 | [−6.81, +3.55] |
| aihub_news | **+16.14** | [+12.31, +20.11] | **+13.95** | [+10.18, +17.61] | **−6.71** | [−10.60, −2.99] |
| naver_news | **+27.83** | [+23.43, +32.18] | **+27.17** | [+23.05, +31.13] | +4.34 | [−1.45, +9.95] |

**파인튜닝 효과는 분명하다.** 베이스 zero-shot 대비 전체 +20.38, 네 도메인 모두 신뢰구간이 0 밖이다. 이 격차는 길이 아티팩트가 아니다 — 베이스는 같은 프롬프트에서 정답보다 2.12배 길게 쓰는데, 프롬프트로 길이를 지시해 1.29배까지 줄여도 격차는 +19.20으로 거의 그대로다.

**반면 lead-3(복사) 대비는 도메인별로 갈린다.** 전체 +0.23은 서로 반대 방향인 도메인이 상쇄된 값이다. 사설에서는 유의하게 낫고(+5.28), AI Hub 신문기사에서는 유의하게 나쁘다(−6.71). naver 뉴스 +4.34와 법률 −1.66은 이 표본 크기에서 잡음과 구별되지 않는다.

**두 결론이 충돌하는 게 아니다.** 어댑터가 배운 것의 큰 부분은 "정답 요약의 어휘·길이·문체에 맞추는 것"이고, 정답 요약 자체가 본문 어휘를 많이 재사용하는 도메인(법률·AI Hub 신문기사)에서는 그 상한이 복사 베이스라인이다.

**테스트 결과 요약**

- **성능**: 단순 본문 복사기 수준을 벗어나 문장을 새로 구성하는 추상적 요약 능력을 확보.
- **도메인별 특성**: 사설 및 일반 뉴스 영역에서는 Lead-3 베이스라인 대비 유의미한 성능 향상을 달성.
- **한계**: 법률·AI Hub 신문기사처럼 정답 요약이 본문 어휘를 그대로 많이 쓰는 도메인에서는 Lead-3를 확실히 이기지 못했다 — "파인튜닝이 항상 복사보다 낫다"는 결론은 과장이며, 도메인 특성에 좌우된다.

### 5. Future work 및 팀 회고

**Future work**
- RAG를 메인 다이제스트 파이프라인에 실제 연결 (현재는 DB로만 연결된 독립 서브시스템)
- 골드셋 확장 —  데이터를 수백 건 규모로 늘리고, 정식 크기 모델(7~8B)로 전환하는 게 다음 단계
- GLiNER2 이벤트/관계 추출의 환각 필터링 — 3.2절에서 발견한 오류 패턴(사실 왜곡, 역할 혼동, 과다 생성)에 대한 후처리 로직 설계
- 프롬프트 개선 3라운드 루프(3.8절)에서 “안전 우선” 지시가 역효과였다는 게 확인됐으니, 다음은 프롬프트가 아니라 **학습 데이터에 애매한 경계 사례를 더 채워 재학습**하는 방향으로 진행

**KPT**

| Keep | Problem | Try |
| --- | --- | --- |
| 집계 지표만 믿지 않고 개별 쿼리로 파고들어 리랭커 착시를 걷어냄 | 실제 학습은 진행됐는데 WORKLOG/registry에 기록이 안 남아 “실행 이력 있는지”조차 코드만 봐선 판단 불가 | 실행 즉시 WORKLOG 기록하는 습관화 |
| `pipeline_status` 기반 재시작 설계 | 필요한 순간에야 발견되는 설정 누락(`HF_TOKEN` 등)·의존성 누락(GLiNER2, bs4) | `.env.example` ↔︎ 코드 변수명 자동 검증, 서브패키지별 환경 점검 스크립트 |
| 골드셋 생성 과정에서 알고리즘 판단 오류 2건, 학습 코드 버그 3건을 직접 실행하며 발견 | 검색 테스트가 “DB가 비어있다”는 전제에 기대고 있어, 실 데이터 인덱싱 후 3건 깨짐 | 테스트를 fixture 스코프로 격리 |
| 결과가 안 좋을 때 모델을 키우는 대신 데이터 자체(클래스 불균형)를 먼저 의심 | 외부 API(네이버)가 최근 NAVER API HUB로 이관되며 엔드포인트·인증 헤더가 바뀐 걸 문서 없이는 알 수 없었음(curl로 직접 원인 특정) | 서드파티 API 연동 코드에 “최근 확인일”을 주석으로 남겨, 이관·정책 변경을 더 빨리 의심할 수 있게 |
| “프롬프트를 더 정교하게 만들수록 좋아진다”는 가정을 실제 3라운드 실험으로 반증 — 베이스라인이 제일 균형 잡혔음을 확인 | “애매하면 안전한 선택을 하라”는 프롬프트 지시가 데이터 불균형과 똑같은 클래스 붕괴를 일으킴(R3) | 프롬프트 수정보다 학습 데이터의 경계 사례 보강을 우선 |
| qlora 적용하여 finetuning 걸리는 시간 감소 | 한번 테스트해보고 결과보는데 너무 시간이 오래 걸려서 힘들고 성능을 향상 시키는 과정이 어렵다. | 모델 성능 추가 고도화 필요로함 |
| 데이터 중복 제거 및 랜덤 샘플링기능 개선 | 공용 데이터를 그냥 사용하니 테스트와 학습데이터 에 데이터 중복으로 누수가 발생하는 문제 | 데이터의 무결성을 지키며 중복 제거를 위한 알고리즘 구상 |
|  |  |  |

### 기타. 레퍼런스

AI프로젝트를 위한 - CrewAI로 만드는 뉴스 브리핑 자동화 에이전트

n8n으로 뉴스 요약 자동화 만들기

매일 아침 AI 뉴스 리포트 자동 생성하기: Claude Cowork

News research and sentiment analysis AI agent with Gemini and SearXNG


# news-insight-agent (MVP 고정 파이프라인)

GNews 수집 → SQLite 저장 → 요약 → 인사이트/비즈니스 시사점 → 이메일 발송까지, 조건 분기 없이 고정된 순서로 실행되는 LangGraph 파이프라인이다. `mvp-implementation-breakdown.md`의 1단계(#1~#8) 구현체다.

## 구조

```
news-insight-agent/
  config.py              # .env 로드
  main.py                 # CLI 진입점 (#7 실행 트리거, #8 스케줄 대상)
  Dockerfile.db           # PostgreSQL 이미지 (스키마 초기화 포함)
  scripts/
    db-up.ps1 / db-up.sh  # DB 컨테이너 기동 스크립트
  db/
    schema.sql            # 테이블 정의 (PostgreSQL)
    db.py                  # PostgreSQL 헬퍼 (#2)
  tools/
    news_fetch.py          # GNews 수집 (#1)
    llm_client.py           # Anthropic 공용 호출
    summarize.py            # 요약 (#3)
    insight.py               # 인사이트+시사점 (#4)
    email_format.py          # Jinja2 렌더링 (#5)
    email_send.py            # Resend 발송 (#6)
  templates/
    digest_email.html.j2     # 이메일 템플릿
  graph/
    pipeline.py               # LangGraph StateGraph (#7)
  trigger/
    scheduler.py              # 간격 기반 트리거 스케줄러 (설계문서 3.1)
    jobs.py                   # 트리거가 실행할 작업 (현재는 hello_world)
    __main__.py               # 트리거 CLI
  tests/
    test_scheduler.py         # 트리거 테스트
  .github/workflows/
    daily-digest.yml          # 스케줄 실행 예시 (#8)
```

## 설정

1. `.env`에 `GNEWS_API_KEY`, `ANTHROPIC_API_KEY`, `RESEND_API_KEY`, `EMAIL_FROM`, `EMAIL_TO`를 채운다.
2. `pip install -r requirements.txt`
3. DB(PostgreSQL) 컨테이너를 띄운다.

```powershell
.\scripts\db-up.ps1          # Windows
```

```bash
./scripts/db-up.sh           # macOS / Linux
```

## DB (PostgreSQL 컨테이너)

기사·다이제스트·발송 이력은 `docker-compose.yml`의 `db` 서비스(PostgreSQL 16)에 저장한다.
이미지는 `Dockerfile.db`가 공식 `postgres:16-alpine`에 `db/schema.sql`을
`/docker-entrypoint-initdb.d`로 얹은 것이고, 데이터는 `pgdata` 볼륨에 영속화된다.

```powershell
.\scripts\db-up.ps1                # 빌드 + 기동 + 준비 대기 + 접속 정보 출력
.\scripts\db-up.ps1 -Psql          # 기동 후 psql 세션
.\scripts\db-up.ps1 -Logs          # 기동 후 로그 팔로우
.\scripts\db-up.ps1 -Down          # 정지 (데이터 유지)
.\scripts\db-up.ps1 -Reset -Force  # 데이터 볼륨까지 삭제하고 재생성
```

`db-up.sh`도 같은 기능을 `--psql`, `--logs`, `--down`, `--reset --force`로 제공한다.

접속 정보는 `DATABASE_URL` 하나로 결정된다. 없으면 `POSTGRES_HOST/PORT/DB/USER/PASSWORD`로
조립하며 기본값은 `postgresql://briefyi:briefyi@localhost:5432/briefyi`다. 컨테이너 안에서
실행할 때는 호스트가 `localhost`가 아니라 서비스명 `db`이며, 이 값은 `docker-compose.yml`이
주입한다.

| 환경변수 | 기본값 | 설명 |
| --- | --- | --- |
| `DATABASE_URL` | 아래 값들로 조립 | 접속 문자열. 관리형 DB를 쓸 때 이 값만 주면 된다 |
| `POSTGRES_HOST` / `POSTGRES_PORT` | `localhost` / `5432` | 호스트에서 접속할 주소 |
| `POSTGRES_DB` / `POSTGRES_USER` / `POSTGRES_PASSWORD` | `briefyi` | DB 이름·계정. 컨테이너 초기화에도 같은 값이 쓰인다 |

스키마는 컨테이너 최초 기동 시 자동 적용되고, 앱 시작 시 `init_db()`가 `CREATE TABLE IF NOT
EXISTS`로 한 번 더 확인한다. 기존 볼륨이 있는 상태에서 스키마를 바꿨다면 `-Reset -Force`로
볼륨을 지우거나 마이그레이션을 직접 적용해야 한다.

계정 정보, 테이블별 컬럼/제약, 자주 쓰는 쿼리, 트러블슈팅은 `docs/database.md`에 정리해 두었다.

## 실행

두 가지 모드가 있다. 기본은 `single`이며 `RUN_MODE` 환경변수로 기본값을 바꿀 수 있다.

```bash
# single: 1회 실행 후 종료 (cron/GitHub Actions 등 외부 스케줄러가 정기 호출)
python main.py --keyword "AI" --days 1 --max-results 10

# trigger: 프로세스를 띄워둔 채 주기 반복 실행 (Ctrl+C 종료)
python main.py --mode trigger --interval 3600
python main.py --mode trigger --interval 10 --duration 60   # 60초만 돌고 종료
```

| 옵션 | 기본값 | 설명 |
| --- | --- | --- |
| `--mode` | `RUN_MODE` (single) | `single`=1회 실행, `trigger`=주기 반복 |
| `--interval` | `TRIGGER_INTERVAL_SECONDS` (86400) | trigger 모드 실행 주기(초) |
| `--duration` | 없음 | trigger 모드에서 이 시간(초) 뒤 자동 종료 |
| `--keyword` / `--days` / `--max-results` | `.env` 값 | 파이프라인 파라미터 (두 모드 공통) |

모드별 종료 동작이 다르다. `single`은 파이프라인이 실패하면 종료 코드 1을 반환하므로 cron/CI가
실패를 감지할 수 있다. `trigger`는 한 주기의 실패를 로그와 실패 카운트로만 남기고 다음 주기를
계속 돌며, 종료 시 성공/실패 횟수를 요약 출력한다.

실행 전에 DB 컨테이너가 떠 있어야 한다(위 "DB" 섹션). 첫 실행 시 테이블이 없으면
`init_db()`가 생성한다.

## 파이프라인 흐름 (고정 순서, 분기 없음)

`fetch_news` → `store_raw` → `summarize` → `extract_insight` → `format_email` → `send_email`

각 노드는 `graph/pipeline.py`의 `PipelineState`(TypedDict)를 입출력으로 공유한다. 노드 하나가 실패하면 `error` 필드에 기록되고 `main.py`가 비정상 종료 코드를 반환한다.

## 스케줄링

`.github/workflows/daily-digest.yml`은 매일 08:00 KST에 `main.py`를 실행하는 GitHub Actions 예시다. 리포지토리 Secrets에 API 키를 등록하면 그대로 동작한다. 서버에서 직접 돌린다면 동일한 명령을 cron에 등록하면 된다.

```
0 8 * * * cd /path/to/news-insight-agent && /usr/bin/python3 main.py >> logs/run.log 2>&1
```

## 트리거 (프로세스 내 주기 실행)

cron/GitHub Actions 없이 프로세스 안에서 주기 실행이 필요할 때 쓰는 트리거 계층이다
(설계문서 3.1). 표준 라이브러리만 사용하며 `main.py --mode trigger`가 이 스케줄러를 쓴다.

트리거 자체를 직접 돌릴 수도 있다. `trigger/jobs.py`의 `JOBS`에 등록된 작업을 이름으로
지정한다 (`hello`=동작 확인용 hello_world, `digest`=`main.run_digest` 호출).

```bash
python -m trigger                          # hello를 10초마다 실행 (Ctrl+C 종료)
python -m trigger --interval 5             # 5초 주기
python -m trigger --duration 25            # 25초만 돌고 종료
python -m trigger --once                   # 1회만 실행
python -m trigger --job digest --interval 3600   # 파이프라인을 1시간 주기로
```

특성:

- **고정 주기**: 작업 소요 시간과 무관하게 `이전 예정 시각 + interval`로 다음 실행을 잡는다.
  10초 주기면 작업이 1초 걸려도 10초마다 실행된다. 한 주기를 넘길 만큼 늦어지면 밀린 실행은
  몰아서 따라잡지 않고 건너뛴다.
- **단일 스레드 순차 실행**: 긴 작업은 다른 작업을 지연시키므로 무거운 파이프라인은 주기를
  넉넉히 잡는다.
- **작업 예외 격리**: 작업이 던진 예외는 로그로 남기고 `Job.error_count`에 집계하며, 트리거는
  다음 주기를 계속 돈다.

새 작업을 붙이려면 `trigger/jobs.py`에 함수를 추가하고 `JOBS`에 등록하면 `--job <이름>`으로
바로 쓸 수 있다. 코드에서 직접 쓸 때는 다음과 같다.

```python
from trigger import Scheduler
from trigger.jobs import hello_world

scheduler = Scheduler()
scheduler.add_job(hello_world, interval=10.0)
scheduler.run_forever()   # 또는 start() / stop()
```

## 테스트

```bash
python -m unittest discover -s tests -t .                  # 전체 (빠름, 1초 미만)
python -m unittest tests.test_scheduler                    # 스케줄러
python -m unittest tests.test_main_modes                   # main의 single/trigger 모드
python -m unittest tests.test_db_connection                # DB 접속 (URL 조립 + 실접속)
python -m unittest tests.test_db_crud                      # 테이블별 CRUD·제약·트랜잭션
python -m unittest tests.test_db                           # db.py 헬퍼 함수
RUN_SLOW_TESTS=1 python -m unittest tests.test_scheduler   # 실제 10초 주기 검증 포함 (약 21초)
```

빠른 테스트는 가짜 시계를 주입해 실제 대기 없이 10초 주기·지연 시 건너뛰기·예외 격리·다중
작업 주기를 검증한다. `test_main_modes`는 `run_pipeline`/`init_db`를 mock으로 대체해 외부
API 호출이나 이메일 발송 없이 모드별 동작(종료 코드, 주기 반복, 실패 후 계속 실행)을 확인한다.
실제 시간으로 10초 간격을 재는 통합 테스트는 시간이 걸리므로 `RUN_SLOW_TESTS=1`일 때만 실행된다.
`test_db*`는 실제 PostgreSQL에 붙는 통합 테스트로, DB에 접속할 수 없으면 skip된다. 테스트가
만든 행은 tearDown에서 삭제한다(공용 헬퍼: `tests/dbhelpers.py`).

## 다음 확장 (백로그, `mvp-implementation-breakdown.md` 2단계 참고)

Discord 발송(#9), 검증/품질게이트(#10), 기술문서 수집(#11), 중복제거/클러스터링(#12), 실시간 트리거(#13)는 아직 구현하지 않았다. `tools/`에 새 도구를 추가하고 `graph/pipeline.py`에 노드/엣지를 붙이는 방식으로 확장하면 된다. 예를 들어 Discord는 `tools/discord_send.py`를 만들고 `format_email` 다음에 `send_discord` 노드를 병렬로 붙이면 된다. 검증 게이트를 추가할 때 비로소 `add_conditional_edges`로 "재작업 여부"를 LLM이 판단하게 해, 고정 파이프라인에서 에이전틱 파이프라인으로 전환할 수 있다.

## 참고 문서

- `docs/database.md`: DB 접속 정보·테이블 구조·쿼리 모음
- `docs/docker.md`: 컨테이너 빌드/실행/영속화

- GNews Search Endpoint: https://docs.gnews.io/endpoints/search-endpoint
- LangGraph StateGraph: https://reference.langchain.com/python/langgraph/graph/state/StateGraph
- Resend API: https://resend.com/docs/api-reference/emails/send-email
