# CLAUDE.md

Bu dosya, Claude Code'un bu projede çalışırken otomatik okuduğu bağlam
dosyasıdır. Amaç: projeyi baştan anlatmak zorunda kalmadan, geçmişteki
tasarım kararlarını ve bilinen sınırları Claude Code'a aktarmak.

## Proje nedir

Kurumsal bir RAG asistanının **çıktılarını** güvenilirlik açısından ölçen,
iç ağda çalışacak şekilde tasarlanmış bir değerlendirme platformu. Model
kodunu değil, modelin ürettiği cevapları denetler. Yedi boyutta ölçüm
yapıp 0-100 arası tek bir Trust Score üretir.

## Kritik geçmiş — bunları tekrar önermeye çalışma

**Track A (kod güvenilirliği hattı) kasıtlı olarak kaldırıldı.** Proje
başta iki paralel hat olarak kurgulandı: kod analizi (Pylint/Bandit/sandbox)
ve LLM çıktı analizi. Staj görevlisinin geri bildirimi üzerine kod hattı
tamamen silindi çünkü kurumun önceliği asistanın *ne söylediği*, AI'ın
*nasıl kod yazdığı* değil. Eğer bir görev "kod kalitesi de eklensin" gibi
bir şey istemiyorsa, bu konuyu yeniden gündeme getirme.

**"EU AI Act uyumu" iddiası bilerek terk edildi.** İlk planda en yüksek
öncelik EU AI Act'e uyumdu. Araştırma sonucu bu yanlış çıktı: Madde 2(3)
askeri/savunma amaçlı sistemleri kapsam dışı bırakıyor, ayrıca şirket AB
üyesi olmayan bir ülkede. Bunun yerine **NIST AI RMF** ve **ISO/IEC
42001** (gönüllü, coğrafyadan bağımsız) birincil çapa yapıldı; EU AI Act
yalnızca "bağlayıcı olmasa da iyi tasarım referansı" olarak ikincil
konumda tutuluyor. Detay: `docs/GOVERNANCE_ALIGNMENT.md`. Bu ayrımı asla
gevşetme — "AI Act'e tabiyiz" gibi bir cümle yazma.

**Ağırlıklar "ölçülmedi", türetildi.** ISO/IEC 25010'un 9 karakteristiğinden
3'ü (Safety, Security, Functional suitability) bu proje için seçildi ve
aralarına eşit ağırlık verildi — bu standardın kendi emri değil, projenin
kararı. Security içindeki alt bölüşüm (injection:pii:poisoning = 3:2:1)
OWASP LLM Top 10 sıralamasından rank-sum yöntemiyle geldi. Bkz.
`docs/METHODOLOGY.md`. Sakın "ağırlıkları ölçtük" diye yazma; "türettik"
veya "belgelenmiş bir yöntemle seçtik" de.

**İnsan kalibrasyonu gerçek değil, iskelet.** `core/calibration.py`
örnekleme + analiz araçlarını içerir ama gerçek insan etiketleyici verisi
yoktur. `demo` komutu ürettiği her sayıyı `is_synthetic: true` ile
işaretler. Bu modülü asla "kalibre edildi" diye sunma; gerçek bir insan
etiketleme çalışması yapılmadan bu iddia doğru değil.

## Mimari — yedi katman, tek yönlü bağımlılık

```
config/settings.yaml (Pydantic)
  → core/ (schemas, logging, storage, tracking, scoring, trace, audit, versioning)
    → llm_security/, rag/, capability/ (7 boyutun analiz modülleri)
      → core/dimensions.py (kayıt sistemi — boyutlar kendini kaydeder)
        → benchmark_engine.py (orkestrasyon)
          → app.py (Streamlit dashboard, sadece okur)
```

**Boyutlar motora gömülü değil, kayıtlı (registry pattern).** Yeni bir
ölçüm boyutu eklemek `benchmark_engine.py`'yi değiştirmeyi gerektirmez —
`core/dimensions.py` içinde `register(Dimension(...))` çağrısı yeterli.
Bu iddia `tests/test_dimension_registry.py` ile kanıtlanmış durumda; yeni
bir boyut eklersen bu testin mantığını bozma.

## Yedi boyut ve ISO sütunları

| Sütun | Boyut | Ağırlık |
|---|---|---|
| Safety | content_safety | 0.33 |
| Security | injection (OWASP LLM01) | 0.17 |
| Security | pii (OWASP LLM02) | 0.11 |
| Security | poisoning (OWASP LLM04) | 0.06 |
| Functional | retrieval | 0.11 |
| Functional | generation (faithfulness) | 0.11 |
| Functional | math | 0.11 |

## Fallback zincirleri — bilinçli tasarım, "eksik" değil

Ağır bağımlılıklar (chromadb, sentence-transformers, ragas) opsiyoneldir.
Yoklarsa sistem sırasıyla NumPy vektör deposu, hashing embedding, heuristic
faithfulness moduna düşer. Hangisinin kullanıldığı her sonuçta `backend`
alanında raporlanır. Bunu "kırık" sanıp zorla ağır paket kurmaya çalışma;
bu davranış kasıtlı ve iç ağ (offline) senaryosu için gerekli.

## Denetim izi ve versiyonlama

`core/audit.py` hash-zincirli, değiştirilemez bir kayıt tutar (kim, ne
zaman, hangi kod/config sürümü). `core/versioning.py` test verisinin
(korpus, senaryolar, sözlükler) hash'ini alıp "puan farkı veri
değişikliğinden mi model değişikliğinden mi" sorusunu otomatik ayırır.
Bu ikisini kaldırma veya basitleştirme — staj görevlisinin özellikle
istediği "denetlenebilirlik" gereksinimini karşılıyorlar.

## Test paketi

172 test geçiyor, 2'si bilinçli `xfail` (sözlük tabanlı içerik filtresinin
bağlam duyarsızlığı — bilinen ve belgelenmiş bir sınır, "düzeltme" deme).
Test kategorileri: işlevsel (HateCheck yöntemi, karşıt vakalar dahil),
dayanıklılık (ReDoS, log injection, kaynak tüketimi), doğrulayıcı (Luhn,
IBAN, TCKN, matematik denkliği), kayıt sistemi, denetim izi.

```bash
pytest tests/ -q
```

## Sık kullanılan komutlar

```bash
python -m rag.ingest --seed --reset
python -m scripts.generate_llm_outputs
python benchmark_engine.py                    # ana koşu
python benchmark_engine.py --preset owasp_rank  # farklı ağırlık ön ayarı
python -m core.audit verify                   # denetim izi bütünlüğü
python -m core.versioning drift --model gpt4  # drift raporu
streamlit run app.py                          # dashboard (3 sekme)
```

## `config/lexicons/` henüz gerçek veri içermiyor

İçerik güvenliği sözlükleri (`profanity.txt`, `religious_insult.txt` vb.)
mekanizma testi için birkaç örnek terimle dolu, gerçek kurumsal liste
değil. `content_safety=100` gibi bir sonuç gördüğünde bunun "temiz" değil
büyük ölçüde "henüz taranmadı" anlamına gelebileceğini unutma — sonuçta
`inactive_categories` alanı hangi kategorinin boş olduğunu gösterir.

`religious_insult.txt` özellikle hassastır: dine yönelik eleştiri ile
hakaret arasındaki sınır teknik bir karar değildir, kurum tanımlamalıdır.
Bu dosyayı kendi inisiyatifinle doldurma.

## Belgeler (ayrıntı gerekince oku)

- `README.md` — kurulum, kullanım, kurumsal olgunluk katmanı özeti
- `docs/METHODOLOGY.md` — her parametrenin kaynak/madde referansı
- `docs/DEFENSE.md` — her tasarım kararının problem/çözüm/kaynak eşlemesi
- `docs/GOVERNANCE_ALIGNMENT.md` — NIST/ISO/EU AI Act eşlemesi ve neden bu sırayla
- `docs/CALISMA_KAGIDI.md` — sunum için tek dosyalık özet
