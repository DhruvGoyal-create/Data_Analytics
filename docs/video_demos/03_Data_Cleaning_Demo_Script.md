# Video Demonstration & Presentation Script: Foundational Analytics — Task 3

## Title: Deterministic Data Cleaning & Integrity Assurance Pipeline for Cafe Sales
**Candidate:** `Dhruv Goyal` *(Notice: Template script; replace with student name prior to recording)*  
**Track:** Data Analytics  
**Task Code:** `PRJ-03`  
**Video Target Duration:** 3 to 4 Minutes  
**Resolution:** 1080p (1920x1080) at 30/60 fps  

---

### Mandatory Compliance Notice:
> **Official Rule DEC-005 & Quality Gate:** The video demo MUST start with a strict 2-second static title card displaying (1) Full Student Name, (2) Assigned Track, and (3) Task Title. The pre-rendered title card is located at `docs/Video_Demo/Week_01/title_card_task3.png`.  
> **Status:** The title card on disk is currently a **template** with `Dhruv Goyal`. It is **NOT submission-ready** until the candidate renders their actual legal name.

---

## Storyboard & Narration Timeline

```
[00:00 - 00:02] --- STATIC TITLE CARD (Dhruv Goyal | Track: Data Analytics | Task: PRJ-03 Cleaning Data)
[00:02 - 00:30] --- Act 1: Problem Context & Dirty Data Forensics
[00:30 - 01:10] --- Act 2: Baseline Data Quality Audit & Figure 1 Walkthrough
[01:10 - 01:50] --- Act 3: Independent Proof on Directly Observed Records & The Menu Key
[01:50 - 02:25] --- Act 4: Deterministic Algebraic Solving & The 26 Unrecoverable Rows
[02:25 - 03:00] --- Act 5: Preserving Missing Dates vs. Median Imputation Distortion
[03:00 - 03:35] --- Act 6: Statistical Outlier Audit vs. Domain Defense (Figure 2)
[03:35 - 04:00] --- Act 7: Automated Unit Test Suite Execution & Summary
```

---

## Detailed Narration Script

### Scene 1: Static Title Card (00:00 – 00:02)
- **Visual:** Display static full-screen image `title_card_task3.png` for exactly 2 seconds.
- **Audio:** *(Silence or low intro chime)*.

---

### Scene 2: Project Overview & The Real-World Data Problem (00:02 – 00:30)
- **Visual:** Open VS Code showing `projects/03_Data_Cleaning_Pipeline/README.md` and `Data_Cleaning_Pipeline.ipynb`.
- **Narration:**
  > "Hello everyone, my name is Dhruv Goyal, and this is my presentation for Project 03 (Data Cleaning Pipeline) of the DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO: Data Cleaning.
  > 
  > In real-world enterprise environments, data rarely arrives neatly formatted. Point-of-sale systems and database migrations frequently introduce placeholder strings like `'UNKNOWN'` and `'ERROR'`, collapse numerical fields into generic text, and create multi-channel missingness. Today, I'll walk you through a production-grade, mathematically verified data cleaning pipeline built using Python, pandas, and NumPy that reconstructs corrupted cafe sales records with 100% mathematical integrity."

---

### Scene 3: Baseline Data Quality Audit & Figure 1 (00:30 – 01:10)
- **Visual:** Scroll to Section 1 of `Data_Cleaning_Pipeline.ipynb` and display `fig1_data_quality_anomalies.png`.
- **Narration:**
  > "We begin with a forensic data quality audit of the raw 10,000-transaction dataset. Notice here in Figure 1 that data corruption was not limited to empty cells:
  > - Location was missing or corrupted in nearly 40% of records.
  > - Payment Method was corrupted in nearly 32% of records.
  > - Core financial fields—Item, Quantity, Price, and Total Spent—each had between 5% and 10% corruption rates involving string tokens like `'ERROR'` and `'UNKNOWN'`.
  > - Furthermore, all eight features initially loaded as generic text objects.
  > 
  > We also audited duplicate rows: while all 10,000 Transaction IDs are completely unique, we identified 141 natural attribute collisions. Because multiple customers naturally purchase the same item on the same day in a cafe, these are valid business events, not duplicate defects."

---

### Scene 4: Independent Proof on Directly Observed Records & The Menu Key (01:10 – 01:50)
- **Visual:** Show Section 2 where the 8,544 directly observed records are checked.
- **Narration:**
  > "Before using formulas to reconstruct missing numbers, we verified our assumptions against reality. On the 8,544 rows where Quantity, Price, and Total Spent were all directly observed in raw data, the identity Total Spent equals Quantity times Price held for 100.00% of rows with zero mismatches. This proves the data generation process follows pure multiplication without hidden fees.
  > 
  > Furthermore, by inspecting uncorrupted records, we confirmed fixed canonical menu prices, from $1.00 for Cookies to $5.00 for Salads."

---

### Scene 5: Deterministic Solving & The 26 Unrecoverable Rows (01:50 – 02:25)
- **Visual:** Show deterministic solving and the unrecoverable purge table.
- **Narration:**
  > "Using this structure, we solved for missing values algebraically:
  > - Quantity was reconstructed using integer division with floating-point tolerance (< 1e-5).
  > - Price was back-calculated or looked up from the menu.
  > - Unique price points like $1.00 and $5.00 were reverse-mapped to item names.
  > 
  > Following reconstruction, exactly 26 rows out of 10,000 remained unresolvable because two or more core financial dimensions were missing simultaneously. Rather than fabricating data, we safely purged these 26 records—representing only 0.26% of transactions and $54 in revenue—allowing us to retain 9,974 records with 100% mathematical integrity."

---

### Scene 6: Preserving Missing Dates vs. Median Imputation (02:25 – 03:00)
- **Visual:** Show Section 3 date handling discussion.
- **Narration:**
  > "A crucial engineering decision was date handling. 460 transactions had missing or unparseable dates. A naive approach would fill them with the median date merely to achieve 'zero nulls'. But doing so would artificially inject 460 transactions into a single summer day!
  > 
  > Instead, we strictly preserved these 460 dates as missing—NaT in memory and blank in CSV. They remain fully valid for overall financial and product analysis, while being properly excluded from temporal time-series charts."

---

### Scene 7: Statistical Outlier Audit vs. Domain Defense (03:00 – 03:35)
- **Visual:** Scroll to Section 4 of the notebook and display `fig2_outlier_detection.png`.
- **Narration:**
  > "Next, we audited outliers using IQR and Z-score methods. In Figure 2, you can see that Quantity and Price have zero outliers, but Total Spent has 259 records flagged above the $24.00 threshold, all landing at exactly $25.00.
  > 
  > Every single one of these 259 transactions is a customer purchasing 5 Salads at $5.00 each. Capping them would have broken our accounting formula and destroyed $259 in valid revenue. We successfully defended retaining 100% of these records as legitimate business transactions."

---

### Scene 8: Automated Unit Tests & Wrap-Up (03:35 – 04:00)
- **Visual:** Switch to terminal and run `python -m unittest test_cleaning_pipeline.py`.
- **Narration:**
  > "Finally, we verify everything with an automated 9-point unit test suite. Running `unittest` confirms that schema types, integer quantity reconstruction, date preservation, and accounting identities all pass with zero errors in under a tenth of a second.
  > 
  > The clean dataset is now exported to `data/cleaned_cafe_sales.csv`. Thank you for watching!"
