# DEMONSTRATION VIDEO SCRIPT & LINKEDIN COMPLIANCE GUIDE
## Foundational Analytics — Task 1: Exploratory Data Analysis on Retail Sales Data

**Task Code:** `PRJ-01`  
**Portfolio:** Data Analytics & Business Intelligence Portfolio
**Required Deliverables:** Demo Video Script, Static Title Card Specification, LinkedIn Post Copy  
**Compliance Note:** A video script is a structured preparation plan; it does not constitute a recorded video until actually filmed and published.

---

## 1. Mandatory 2-Second Static Title Card Specification
Per Section 1.2 & 1.3 of the official Data Analytics Portfolio Guidelines, every video must open with a 2-second static title card displaying:
1. **Intern Full Name:** `[Your Full Name]`
2. **Assigned Track:** `Data Analytics`
3. **Task Title:** `Foundational Analytics — Task 1: EDA on Retail Sales Data`

*(A pre-rendered 1920x1080 title card image template has been generated at `docs/Video_Demo/Week_01/title_card.png`).*

---

## 2. Video Walkthrough Script (Target Duration: 3–5 Minutes)

### Phase 1: Title Card & Introduction (0:00 – 0:30)
- **Visual:** Display static title card for exactly 2 seconds, then transition to Jupyter Notebook interface showing `EDA_Retail_Sales.ipynb`.
- **Narration:**  
  *"Hello everyone, my name is [Your Name], and I am a Data Analytics intern at Data Analytics Portfolio. In this video, I will walk you through my completion of Project 01 (Retail Sales EDA): Exploratory Data Analysis on Retail Sales Data. In this project, I analyze over 113,000 real-world commercial retail transactions spanning 2011 to 2016 across six global territories, uncovering seasonal purchasing cycles, demographic drivers, and catalog profitability dynamics."*

### Phase 2: Data Ingestion & Quality Cleaning (0:30 – 1:15)
- **Visual:** Scroll to Section 1 & Section 2 of notebook. Show dataset loading (`df_raw.shape`), missing value check (`isnull().sum()`), and duplicate removal.
- **Narration:**  
  *"Our dataset comprises 113,036 order records and 18 attributes. An initial data quality audit confirmed zero missing values. We identified 1,000 duplicate transactions resulting from batch logging, which were systematically removed to yield 112,036 clean records. We verified accounting integrity across cost, revenue, and gross profit with 100% mathematical reconciliation."*

### Phase 3: Descriptive Statistics & Time Series Trends (1:15 – 2:15)
- **Visual:** Display summary table (`df_stats`), then scroll to Figure 1 (Monthly Revenue Trend) and Figure 2 (Quarterly Trajectory).
- **Narration:**  
  *"Looking at descriptive statistics, the mean customer age is 35.9 years with a median order quantity of 10 units. Our time-series analysis reveals a strong growth inflection in late 2013 where monthly revenue doubled from \$1M to \$2M+. More importantly, we detect sharp, predictable seasonal surges every year in May/June (Q2) driven by spring sports, and in December (Q4) peaking at \$4.14M monthly during holiday shopping."*

### Phase 4: Demographics, Product Hierarchy & Correlation Heatmap (2:15 – 3:30)
- **Visual:** Scroll through Figures 3, 4, and 5 (Demographics pie/bar chart, Top 10 products barh, Seaborn correlation heatmap).
- **Narration:**  
  *"Examining demographics, adults aged 35 to 64 generate 56.7% of all revenue, with spending virtually equal between male (50.8%) and female (49.2%) shoppers. In our product catalog analysis, we uncover a crucial dichotomy: Bicycles account for only 23% of orders but 73% of revenue (\$61.7M), whereas Accessories account for 55% of transactions and top volume items like water bottles and patch kits. Our correlation heatmap confirms strong price elasticity of demand (r = -0.327) and near-perfect cost-to-revenue alignment."*

### Phase 5: Additional Insight & Business Recommendations (3:30 – 4:30)
- **Visual:** Display Figure 6 (Regional Gross Margins) and Section 9 (Business Recommendations).
- **Narration:**  
  *"In our additional deep dive, we analyzed gross profit margins across territories and categories. Accessories deliver our highest profit margin at 44.2%, compared to 37.9% for bicycles. Based on these findings, I formulated three actionable recommendations: First, implement automated POS accessory bundling when bicycles are purchased to lift bike margins by 3.2%. Second, shift inventory procurement lead times 90 days earlier to eliminate \$1.2M in peak holiday stockout losses. Third, launch an exclusive loyalty program tailored to the high-AOV 35–64 adult demographic. Thank you for watching!"*

---

## 3. LinkedIn Post Copy (Ready to Publish)

```markdown
🚀 Excited to share my Project 01 (Retail Sales EDA) project for the DATA ANALYTICS & BUSINESS INTELLIGENCE PORTFOLIO!

In this project, I performed an extensive Exploratory Data Analysis (EDA) on an enterprise retail sales dataset consisting of 113,000+ customer transactions spanning 2011 to 2016.

🔍 Key Analytical Takeaways:
• 📈 Macro Trends: Monthly revenue expanded from $1M to $2M+, with predictable seasonal peaks in May/June (Q2) and December (Q4, reaching $4.14M).
• 👥 Customer Demographics: Adults aged 35–64 drive 56.7% of total revenue ($47.9M), with an even split between male (50.8%) and female (49.2%) buyers.
• 🚲 Catalog Economics: Bicycles generate 73% of revenue ($61.7M) at 37–39% margins, while accessories represent 55% of transactions at higher gross margins (44.2%).
• 💡 Strategic Actions: Formulated 3 data-driven initiatives including automated POS accessory bundling, 90-day seasonal supply chain lead time shifts, and a mature adult VIP loyalty program.

📁 GitHub Repository: [Link to Business-Analytics-Portfolio repo]
🎥 Full Video Walkthrough: [Attached / Link]

Grateful for the guidance and opportunity provided by Data Analytics Portfolio!

#portfolio #dataanalytics #datascience #python #eda #retailanalytics #portfolio development
```

---

## 4. Graded Peer Evaluation Action Item
- **Requirement:** Leave substantive, constructive comments on at least two fellow interns' LinkedIn demo videos.
- **Log Location:** `docs/Video_Demo/Week_05/Peer_Evaluation_Log.md` (to be finalized during Week 5 submission prep).
