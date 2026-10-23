# Extensible datasets — teslim doğrulaması

Onaylı plan: `plans/2026-10-16-extensible-datasets.md`.

- Ortam: mevcut `masterthesis` conda Python; yeni ortam/paket kurulmadı.
- Son suite: **200 test, 45.721 saniye, OK**.
- `git diff --check`: exit0.
- Arayüz: geçici veri kaydı → restart olmadan dört baseline ve iki KD route'u.
- Gerçek model kabiliyeti: kayıtlı20 model, `pretrained=False`, CPU dummy forward.
- Eğitim giriş noktaları: dört baseline dry-run ve üç KD türünde single/multi
  train/backward + validation batch; modeller küçük test ağlarıyla değiştirildi.
- Geçici bir KD epoch'u: checkpoint/history/summary → v2 paket → import →
  karşılaştırma → arşiv; ham veri/registry korunması doğrulandı.
- Mount notebook hücresi farklı geçici root'ta çalıştırıldı; aynı dataset_ref
  korundu, değişen byte reddedildi. Ağ/GPU/credential kullanılmadı.
- Bağımsız review bulguları düzeltildi: parametre logging listesi, Windows uzun
  run yolları, tam JPEG decode, üst layout junction, partial teacher, eski
  split çelişkisi, v2 artifact kimliği ve broad run-dir kaldırma sınırları.
- Canlı desktop/mobile inceleme: mevcut native Streamlit tasarımı korundu;
  sınıf listesi ayrıntıya alındı, dataset/teacher/config sırası doğrulandı.
  Ekranlar `.impeccable/review/extensible-datasets/` altında.
- Geçici8502 sunucusu kapatıldı; kullanıcının8501 süreci kapatılmadı.
  Import önbelleği nedeniyle mevcut uygulama bir kez yeniden başlatılmalıdır.
- Graphify kod grafı1639node/3597edge. `graph.html`, `graph.json` ve
  `GRAPH_REPORT.md` güncel; dışlanan raw/output köklerinden node yok.
 94JSON dosyası AST sembolü üretmedi; doküman/görsel semantik indeksleme yapılmadı.
  Tanısal kontrolde eksik/dangling/collapsed/self-loop edge yoktu.

Canlı Colab/Kaggle oturumu, gerçek GPU bellek/süre, bilimsel başarım ve uzak
commit'in yeni API'yi içermesi test edilmedi. Notebook indirmeyi dirty checkout
nedeniyle engelleyen yeni bir kural eklenmedi; yayımlanmamış yeni kodu uzak
ortamda çalıştırmak yine commit/push gerektirir ve bu işlem yapılmadı.
Eski araştırma verileri, sonuçları ve kullanıcı değişiklikleri korunmuştur.
