# Product Recommendation System — Mësimit Makinor

Projekt për analizën e produkteve Amazon me **klasifikim**, **rrjet neuronal**, **grupim (clustering)** dhe **rekomandime produktesh**.

Dataseti: `data/amazon.csv` (~1.465 produkte elektronike nga Amazon India).

---

## Kërkesat

- **Python 3.10+** (rekomandohet 3.11 ose 3.12)
- Windows / macOS / Linux
- Lidhje internet (vetëm për instalimin e bibliotekave)

---

## Konfigurimi

### 1. Klono ose hap projektin

```powershell
cd C:\Users\Admin\Desktop\ProductRecommendationSystem
```

### 2. Krijo mjedis virtual (rekomandohet)

```powershell
python -m venv venv
.\venv\Scripts\Activate.ps1
```

Në macOS/Linux:

```bash
python3 -m venv venv
source venv/bin/activate
```

### 3. Instalo bibliotekat

```powershell
pip install -r requirements.txt
```

### 4. Verifiko datasetin

Sigurohu që ekziston skedari:

```
data/amazon.csv
```

Nëse mungon, vendose datasetin Amazon CSV në folderin `data/`.

---

## Ekzekutimi

Ekzekuto skriptet **nga rrënja e projektit** (ku ndodhet `requirements.txt`).

### Klasifikuesit (5 klasikë + rrjet neuronal)

```powershell
python classifiers_evaluation.py
```

**Çfarë bën:**
- Ndarje train/test (80/20)
- Eksperimente veçorish dhe tuning hiperparametrash
- Vlerësim me accuracy, precision, recall, F1, matrica konfuzioni

**Rezultatet:** `results/`

| Skedar | Përmbajtja |
|--------|------------|
| `comparative_table.csv` | Tabela krahasuese e klasifikuesve |
| `analysis_report.txt` | Raport i plotë |
| `hyperparameter_tuning.json` | Parametrat e testuar |
| `*.png` | Grafikë dhe matrica konfuzioni |

---

### Grupimi (clustering)

```powershell
python clustering_analysis.py
```

**Çfarë bën:**
- Heq etiketat para grupimit
- K-Means dhe **grupim aglomerativ**
- Vizualizime PCA, elbow plot, crosstab
- Krahasim me etiketat reale (ARI, NMI)

**Rezultatet:** `results/clustering/`

| Skedar | Përmbajtja |
|--------|------------|
| `clustering_report.txt` | Raport dhe diskutim |
| `clustering_experiments.csv` | Eksperimentet me parametra |
| `*.png` | Vizualizime |

---

### Rekomandime produktesh

```powershell
python recommendations.py
```

**Çfarë bën:**
- Rekomandon produkte **të ngjashme** (content-based, cosine similarity)
- Rekomandon produkte nga **i njëjti cluster** K-Means
- Liston **top produkte** sipas kategorisë

**Rezultatet:** `results/recommendations.csv`, `results/recommendations_summary.json`

---

### Ekzekuto të gjitha skriptet njëherësh

```powershell
python run_all.py
```

---

## Struktura e projektit

```
ProductRecommendationSystem/
├── data/
│   └── amazon.csv              # Dataseti
├── results/                    # Daljet e klasifikimit
│   └── clustering/             # Daljet e grupimit
├── data_utils.py               # Ngarkim dhe parapërpunim i përbashkët
├── classifiers_evaluation.py   # Klasifikuesit + rrjeti neuronal
├── clustering_analysis.py      # Grupimi
├── recommendations.py          # Rekomandime produktesh
├── run_all.py                  # Ekzekuton të gjitha skriptet
├── requirements.txt            # Varësitë Python
└── README.md                   # Ky skedar
```

---

## Bibliotekat (requirements.txt)

| Biblioteka | Përdorimi |
|------------|-----------|
| `numpy` | Operacione numerike |
| `pandas` | Manipulim të dhënash |
| `scikit-learn` | ML: klasifikim, clustering, metrika |
| `matplotlib` | Grafikë |
| `seaborn` | Vizualizime statistikore |

---

## Shënime

- Ekzekutimi i plotë zgjat ~2–3 minuta (GridSearchCV + rrjeti neuronal).
- Nëse `python` nuk funksionon, provo `py classifiers_evaluation.py`.
- Rezultatet mbishkruhen çdo herë që ekzekuton skriptet.

---

## Detyra e klasifikimit

**Objektivi:** parashikimi i produkteve me vlerësim të lartë (`rating >= 4.0`).

**Veçoritë:** çmimi, zbritja, numri i vlerësimeve, kategoria kryesore.

**Klasifikuesit:** Logistic Regression, Decision Tree, Random Forest, SVM, KNN, MLP (rrjet neuronal).

## Detyra e grupimit

**Etiketat hiqen** para grupimit; përdoren vetëm për vlerësim pas grupimit (ARI, NMI).

**Algoritmet:** K-Means, grupim aglomerativ.