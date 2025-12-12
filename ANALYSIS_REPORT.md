# 📊 E-Ticaret Müşteri Yaşam Boyu Değeri (CLV) Analizi - Detaylı Rapor

**Yazar:** Ege Koçaslan
**Tarih:** 11 Aralık 2025

---

## 🎯 Projenin Amacı ve Çözdüğü Sorunlar

Bu proje, bir e-ticaret şirketinin müşteri verilerini analiz ederek, her müşterinin **yaşam boyu değerini (CLV)** tahmin etmeyi ve müşterileri **değerlerine göre segmentlere ayırmayı** amaçlamaktadır. Bu sayede, pazarlama bütçesini en verimli şekilde kullanmak ve müşteri sadakatini artırmak için stratejik kararlar alınabilir.

### 🔍 Çözülen Temel Sorunlar

1.  **Değerli Müşterileri Belirleme:** Hangi müşterilerin şirkete en çok kazandırdığını ve hangilerinin potansiyel taşıdığını belirlemek.
2.  **Müşteri Kaybını Önleme:** Hangi müşterilerin şirketi terk etme riski altında olduğunu tespit edip, onları geri kazanmak için proaktif adımlar atmak.
3.  **Pazarlama Stratejilerini Kişiselleştirme:** Her müşteri segmentine özel pazarlama kampanyaları (indirimler, VIP programları, sadakat programları) tasarlayarak, pazarlama bütçesini en verimli şekilde kullanmak.
4.  **Gelecek Gelirlerini Tahmin Etme:** Müşterilerin gelecekte ne kadar harcama yapacağını tahmin ederek, şirketin gelir projeksiyonlarını daha doğru bir şekilde yapmak.

---

## 🛠️ Kullanılan Veri Bilimi Teknikleri

Bu projede, ileri seviye veri bilimi ve makine öğrenmesi teknikleri kullanılmıştır:

| Teknik | Açıklama | Neden Kullanıldı? |
| :--- | :--- | :--- |
| **RFM Analizi** | Müşterileri **Recency** (en son ne zaman alışveriş yaptı), **Frequency** (ne sıklıkla alışveriş yapıyor) ve **Monetary** (ne kadar harcıyor) metriklerine göre analiz etme. | Müşteri davranışlarını anlamak ve segmentasyon için temel oluşturmak. |
| **K-Means Clustering** | Müşterileri RFM skorlarına göre **3 ana gruba** (cluster) ayırmak için kullanılan bir **unsupervised machine learning** algoritması. | Müşterileri benzer davranışlarına göre gruplandırmak. |
| **BG/NBD Modeli** | (Beta-Geometric/Negative Binomial Distribution) Müşterilerin gelecekte ne sıklıkla alışveriş yapacağını tahmin etmek için kullanılan bir **olasılıksal model**. | Müşterilerin gelecekteki satın alma davranışlarını tahmin etmek. |
| **Pareto/NBD Modeli** | BG/NBD modeline alternatif olarak, müşteri kaybını da hesaba katan bir başka olasılıksal model. | Model karşılaştırması ve daha doğru tahminler için. |
| **Gamma-Gamma Modeli** | Müşterilerin gelecekteki her bir alışverişinde ne kadar harcayacağını tahmin etmek için kullanılan bir model. | Müşterilerin gelecekteki harcama potansiyelini tahmin etmek. |
| **Veri Görselleştirme** | Matplotlib ve Seaborn kütüphaneleri kullanılarak, analiz sonuçlarını anlaşılır grafiklere dönüştürme. | Karmaşık verileri ve analiz sonuçlarını kolayca anlaşılır hale getirmek. |
| **Yazılım Mühendisliği** | Projeyi modüler bir yapıda (src/, config/, data/, reports/) organize ederek, kodun tekrar kullanılabilirliğini ve sürdürülebilirliğini sağlamak. | Projenin profesyonel ve endüstri standartlarına uygun olmasını sağlamak. |

---

## 📊 Analiz Sonuçları ve Görselleştirmeler

### 1. RFM Dağılımları

Bu grafikler, müşterilerin genel olarak ne kadar süre önce alışveriş yaptığını (Recency), ne sıklıkla alışveriş yaptığını (Frequency) ve ne kadar harcadığını (Monetary) göstermektedir.

![RFM Dağılımları](reports/figures/01_rfm_distribution.png)

**Yorum:**
- **Recency:** Müşterilerin çoğu yakın zamanda alışveriş yapmış, ancak uzun süredir alışveriş yapmayan bir grup da var.
- **Frequency:** Müşterilerin çoğu az sayıda alışveriş yapmış, ancak sık alışveriş yapan küçük bir grup da var.
- **Monetary:** Müşterilerin çoğu düşük miktarlarda harcama yapmış, ancak yüksek harcama yapan küçük bir grup da var.

### 2. Müşteri Segmentleri

Müşteriler, RFM skorlarına göre 8 farklı segmente ayrılmıştır. Bu grafikler, her segmentteki müşteri sayısını ve dağılımını göstermektedir.

![Müşteri Segmentleri](reports/figures/02_rfm_segments.png)

**Yorum:**
- **Loyal Customers (%35.1):** En büyük segment, sadık müşteriler.
- **Champions (%15.3):** En değerli müşteriler, sık ve yüksek harcama yapıyorlar.
- **Lost (%11.2):** Kaybedilmiş müşteriler, uzun süredir alışveriş yapmıyorlar.
- **At Risk (%4.4):** Kaybedilme riski olan müşteriler.

### 3. Segment Karakteristikleri

Bu heatmap, her bir müşteri segmentinin ortalama Recency, Frequency ve Monetary değerlerini göstermektedir.

![Segment Karakteristikleri](reports/figures/03_segment_characteristics.png)

**Yorum:**
- **Champions:** Recency değeri düşük (yeni alışveriş yapmış), Frequency ve Monetary değerleri yüksek.
- **Lost:** Recency değeri çok yüksek (uzun süredir alışveriş yapmamış), Frequency ve Monetary değerleri düşük.
- **At Risk:** Recency değeri yüksek, ancak Frequency ve Monetary değerleri hala iyi.

### 4. RFM Scatter Plot

Bu 3D scatter plot, müşterilerin Recency, Frequency ve Monetary değerlerine göre nasıl dağıldığını göstermektedir. Baloncukların büyüklüğü, harcama miktarını (Monetary) temsil etmektedir.

![RFM Scatter Plot](reports/figures/04_rfm_scatter.png)

**Yorum:**
- Sağ üst köşedeki büyük baloncuklar, en değerli müşterileri (Champions) temsil etmektedir.
- Sol alt köşedeki küçük baloncuklar, daha az değerli veya yeni müşterileri temsil etmektedir.

### 5. K-Means Clusterları

Müşteriler, K-Means algoritması ile 3 ana gruba ayrılmıştır.

![K-Means Clusterları](reports/figures/05_kmeans_clusters.png)

**Yorum:**
- **Cluster 1:** En büyük grup, genellikle sadık ve ortalama müşterileri içerir.
- **Cluster 0 ve 2:** Daha küçük gruplar, genellikle en değerli (Champions) ve en az değerli (Lost) müşterileri içerir.

### 6. Segment Değer Analizi

Bu grafikler, her bir müşteri segmentinin şirkete ne kadar toplam gelir getirdiğini ve her segmentteki bir müşterinin ortalama değerini göstermektedir.

![Segment Değer Analizi](reports/figures/07_segment_value_analysis.png)

**Yorum:**
- **Loyal Customers:** En çok toplam geliri getiren segment.
- **Champions:** Ortalama müşteri değeri en yüksek olan segment.
- Bu, pazarlama bütçesinin en çok bu iki segmente odaklanması gerektiğini göstermektedir.

### 7. İşlem Zaman Çizelgesi

Bu grafikler, şirketin günlük gelir ve işlem sayısındaki trendleri göstermektedir.

![İşlem Zaman Çizelgesi](reports/figures/08_transaction_timeline.png)

**Yorum:**
- Şirketin gelir ve işlem sayısında zamanla bir artış veya azalış olup olmadığı görülebilir.
- Sezonluk etkiler (örneğin, tatil dönemlerinde artış) tespit edilebilir.

---

## 🚀 Stratejik Öneriler

Bu analiz sonuçlarına dayanarak, şirket aşağıdaki stratejik kararları alabilir:

| Segment | Öneri |
| :--- | :--- |
| **Champions** | VIP programları, özel indirimler, yeni ürünlere erken erişim gibi ayrıcalıklar sunarak onları ödüllendirin. |
| **Loyal Customers** | Sadakat programları, kişiselleştirilmiş ürün önerileri ve e-posta pazarlaması ile onları elde tutun. |
| **At Risk** | Geri kazanma kampanyaları (win-back campaigns), özel indirimler ve anketler ile onları geri kazanmaya çalışın. |
| **New Customers** | Hoş geldin kampanyaları, ilk alışveriş indirimleri ve ürün kullanım kılavuzları ile onları eğitin. |
| **Lost** | Neden ayrıldıklarını anlamak için anketler gönderin ve onları geri kazanmak için çok özel teklifler sunun. |

---

## 💡 Sonuç

Bu proje, bir e-ticaret şirketinin müşteri verilerini nasıl analiz edebileceğini ve bu analiz sonuçlarını nasıl stratejik kararlara dönüştürebileceğini göstermektedir. Kullanılan ileri seviye veri bilimi teknikleri, bu projenin **Fortune 500 ve fintech şirketleri için etkileyici bir portföy projesi** olmasını sağlamaktadır.
