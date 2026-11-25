# Video Demonstration & Presentation Script: Foundational Analytics — Task 2

## Title: Behavioral RFM Customer Segmentation & K-Means Clustering on Retail Transactions
**Candidate:** `Dhruv Goyal` *(Notice: Template script; replace with student name prior to recording)*  
**Track:** Data Analytics  
**Task Code:** `PRJ-02`  
**Video Target Duration:** 3 to 4 Minutes  
**Resolution:** 1080p (1920x1080) at 30/60 fps  

---

### Mandatory Compliance Notice:
> **Official Rule DEC-005 & Quality Gate:** The video demo MUST start with a strict 2-second static title card displaying (1) Full Student Name, (2) Assigned Track, and (3) Task Title. The pre-rendered title card is located at `docs/Video_Demo/Week_02/title_card_task2.png`.  
> **Status:** The title card on disk is currently a **template** with `Dhruv Goyal`. It is **NOT submission-ready** until the candidate renders their actual legal name.

---

## Storyboard & Narration Timeline

```
[00:00 - 00:02] --- STATIC TITLE CARD (Dhruv Goyal | Track: Data Analytics | Task: PRJ-02 Customer Segmentation)
[00:02 - 00:30] --- Act 1: Introduction & The Core Business Problem
[00:30 - 01:10] --- Act 2: Transaction Cleaning & RFM Feature Engineering
[01:10 - 01:50] --- Act 3: Extreme Skewness Management & Feature Standardization (Figure 1)
[01:50 - 02:30] --- Act 4: Unsupervised Optimization — Elbow Curve & Silhouette (Figure 2)
[02:30 - 03:15] --- Act 5: Persona Profiles & 2D Cluster Projections (Figures 3 & 4)
[03:15 - 03:45] --- Act 6: Data-Driven CRM Marketing Playbooks
[03:45 - 04:00] --- Act 7: Automated Unit Test Suite & Conclusion
```

---

## Detailed Narration Script

### Scene 1: Static Title Card (00:00 – 00:02)
- **Visual:** Display static full-screen image `title_card_task2.png` for exactly 2 seconds.
- **Audio:** *(Silence or low intro chime)*.

---

### Scene 2: Introduction & The Core Business Problem (00:02 – 00:30)
- **Visual:** Open VS Code showing `projects/02_Customer_Segmentation/README.md` and `Customer_Segmentation.ipynb`.
- **Narration:**
  > "Hello everyone, my name is Dhruv Goyal, and this is my presentation for Project 02 (Customer Segmentation) of the DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO: Customer Segmentation.
  > 
  > In modern retail, treating all customers identically is a recipe for wasted marketing spend and customer churn. Traditional demographic profiles like age and gender don't capture actual purchasing behavior. Today, I'll walk you through an end-to-end behavioral segmentation pipeline using Python, scikit-learn, and the classical Recency, Frequency, Monetary (RFM) framework applied to 541,000 online retail transactions."

---

### Scene 3: Transaction Cleaning & RFM Feature Engineering (00:30 – 01:10)
- **Visual:** Scroll to Section 1 & Section 2 of `Customer_Segmentation.ipynb`.
- **Narration:**
  > "We begin with raw transactional data from the UCI Online Retail dataset. We performed two essential cleaning steps:
  > - First, we filtered out 135,000 unassigned transactions representing guest checkouts, because segmentation requires an identifiable customer entity.
  > - Second, we removed order cancellations and returns to evaluate true completed purchases.
  > 
  > This retained 397,884 clean transactions across 4,338 unique customers. We then engineered customer-level metrics: Recency—days since last order relative to dataset close; Frequency—number of distinct completed orders; and Monetary—total gross customer spend."

---

### Scene 4: Extreme Skewness Management & Scaling (01:10 – 01:50)
- **Visual:** Display `fig1_rfm_distributions.png` comparing raw vs log-transformed distributions.
- **Narration:**
  > "Here in Figure 1, notice the extreme positive right-skewness in raw Frequency and Monetary values. While the median customer spend is £674, the maximum reaches over £280,000 due to wholesale buyers.
  > 
  > Running distance-based algorithms like K-Means on un-transformed data would allow extreme outliers to monopolize clusters. By applying a natural log transformation, followed by `StandardScaler` to achieve zero mean and unit variance, we stabilized the feature space into well-behaved quasi-normal distributions."

---

### Scene 5: Unsupervised Optimization: Elbow Curve & Silhouette (01:50 – 02:30)
- **Visual:** Display `fig2_elbow_and_silhouette.png` showing the dual-axis elbow and silhouette plot.
- **Narration:**
  > "To rigorously select the optimal number of clusters, we evaluated K from 2 to 8 using both Inertia—the sum of squared errors—and the Silhouette Coefficient.
  > 
  > In Figure 2, you can see a distinct elbow inflection at K equals 4, where inertia drops from 6,481 to 3,938. Furthermore, K equals 4 achieves a local silhouette peak of 0.3371. While K equals 2 yields a higher mathematical score, it oversimplifies customers into binary active vs. inactive groups. K equals 4 provides the optimal balance of statistical cohesion and actionable business granularity."

---

### Scene 6: Persona Profiles & 2D Cluster Projections (02:30 – 03:15)
- **Visual:** Display `fig3_cluster_scatter_plots.png` and `fig4_cluster_customer_counts.png`.
- **Narration:**
  > "Here in Figures 3 and 4, we examine our four distinct customer personas:
  > 1. Champions / High-Value VIPs: 716 customers, just 16.5% of the base, but generating nearly 65% of total revenue (£5.78M) with an average spend exceeding £8,000.
  > 2. Loyal & Consistent Regulars: 1,173 customers contributing 24% of revenue, with steady repeat purchases every 70 days.
  > 3. Recent / New Promising Customers: 837 customers who purchased recently within 18 days, but have only 1 to 2 orders.
  > 4. Hibernating / Churned At-Risk Customers: 1,612 customers—37% of the base—inactive for over 6 months."

---

### Scene 7: Data-Driven CRM Marketing Playbooks (03:15 – 03:45)
- **Visual:** Scroll to Section 6 showing the marketing matrix.
- **Narration:**
  > "Each segment receives a tailored CRM marketing playbook:
  > - For Champions: White-glove VIP treatment, personal account managers, and exclusive early access to product launches to ensure maximum retention.
  > - For Regulars: Automated replenishment triggers and cross-sell bundle discounts to elevate them to VIP status.
  > - For New Customers: Onboarding drip campaigns and time-sensitive 15% second-order incentives to build purchasing habituation.
  > - For At-Risk Customers: Aggressive win-back discounts like £20 off, followed by sunsetting unresponsive contacts to protect email deliverability."

---

### Scene 8: Automated Unit Tests & Wrap-Up (03:45 – 04:00)
- **Visual:** Switch to terminal and run `python -m unittest test_customer_segmentation.py`.
- **Narration:**
  > "Finally, we verify the entire pipeline with an automated 8-point unit test suite. Running `unittest` confirms that customer counts, non-null schema constraints, positive physical bounds, and revenue reconciliations pass with zero errors in under a tenth of a second.
  > 
  > The segmented dataset is exported to `data/rfm_customer_segments.csv`. Thank you for watching!"
