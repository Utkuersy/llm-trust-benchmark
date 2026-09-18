# ML Model Yaşam Döngüsü Standardı

Üretime alınan her model için bir Model Kartı hazırlanması zorunludur.
Model Kartı en az şu bölümleri içerir: amaç, eğitim verisi, metrikler,
bilinen sınırlamalar ve etik değerlendirme.

Modeller üretimde 3 ayda bir yeniden değerlendirilir. Performans,
temel çizginin 5 puan altına düşerse model geri çekilir.

Veri kayması (data drift) izleme eşiği, popülasyon kararlılık indeksi
(PSI) için 0.20 olarak belirlenmiştir.
