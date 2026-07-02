# Hillstrom Email Marketing A/B Test

## Introduction

This project analyzes a real-world email marketing A/B test using the Hillstrom MineThatData dataset.

The business question is:

> Should the company send marketing emails, and if yes, which campaign performs best?

The company tested three experiment groups:

1. **Mens E-Mail** — customers received a men's merchandise email
2. **Womens E-Mail** — customers received a women's merchandise email
3. **No E-Mail** — customers received no email and served as the control group

The goal is to measure whether email marketing improves customer behavior and revenue.

---

## Business Problem

Marketing campaigns cost time, money, and customer attention.

A company should not send campaigns just because they can. It should know whether the campaign creates measurable business value.

This project answers:

- Did the email campaigns increase conversion?
- Did the email campaigns increase customer spend?
- Which campaign performed best?
- Which customer segments responded best?
- What should the business do next?

---

## Project Workflow

Business problem  
→ Data source  
→ Data understanding  
→ Experiment design check  
→ Business metric comparison  
→ Hypothesis testing  
→ Revenue lift analysis  
→ Customer segment analysis  
→ Business recommendation  

---

## Dataset

The dataset contains **64,000 customers** from a real email marketing experiment.

Each row represents one customer.

Main columns:

| Column | Meaning |
|---|---|
| `segment` | Experiment group |
| `visit` | Whether the customer visited the website |
| `conversion` | Whether the customer made a purchase |
| `spend` | Amount spent by the customer |
| `history` | Customer purchase history |
| `channel` | Customer channel |
| `zip_code` | Customer zip code type |
| `newbie` | Whether the customer is new |

---

## Tools Used

- Python
- pandas
- NumPy
- SciPy
- Statsmodels
- Matplotlib
- Seaborn
- Jupyter Notebook

---

## Statistical Methods

This project uses:

- A/B testing
- Control vs treatment comparison
- Conversion rate analysis
- Two-proportion z-test
- Revenue lift analysis
- Segment analysis

The two-proportion z-test is used because conversion is a binary outcome:

- `1` = customer converted
- `0` = customer did not convert

---

## Key Results

### Experiment Group Sizes

| Group | Customers |
|---|---:|
| Womens E-Mail | 21,387 |
| Mens E-Mail | 21,307 |
| No E-Mail | 21,306 |

The groups are nearly equal in size, which supports a valid experiment comparison.

---

## Conversion Results

| Group | Conversion Rate |
|---|---:|
| Mens E-Mail | 1.25% |
| Womens E-Mail | 0.88% |
| No E-Mail | 0.57% |

The Mens E-Mail campaign produced the highest conversion rate.

---

## Revenue Lift Results

The Mens E-Mail campaign generated the strongest estimated incremental revenue.

| Campaign | Estimated Incremental Revenue |
|---|---:|
| Mens E-Mail | $16,402.71 |
| Womens E-Mail | About $9,100 |

The Mens E-Mail campaign created the most business value compared with the No E-Mail control group.

---

## Segment Insights

The segment analysis showed:

- Mens E-Mail performed best across all customer channels.
- Multichannel customers had the strongest response.
- Rural customers showed high responsiveness to email.
- Phone customers had lower overall conversion, but still improved with email.
- Web customers responded better to Mens E-Mail than Womens E-Mail or No E-Mail.

The strongest business opportunity appears to be:

> Mens E-Mail campaign sent to Multichannel customers.

---

## Business Recommendation

The company should continue email marketing, but it should prioritize the **Mens E-Mail campaign**.

Recommended strategy:

1. Use Mens E-Mail as the primary campaign.
2. Prioritize Multichannel customers.
3. Prioritize high-value customers with stronger purchase history.
4. Continue testing targeted campaigns instead of sending broad campaigns to everyone.
5. Build a future uplift model to predict which customers are most likely to respond.

The final business decision is:

> Email marketing creates measurable value, but targeted email marketing is likely to produce higher ROI than broad email marketing.

---

## Project Structure

```text
hillstrom-ab-testing/
├── data/
│   ├── raw/
│   └── processed/
├── notebooks/
│   └── 01_data_understanding.ipynb
├── reports/
│   └── figures/
├── src/
│   └── __init__.py
├── .gitignore
├── README.md
└── requirements.txt