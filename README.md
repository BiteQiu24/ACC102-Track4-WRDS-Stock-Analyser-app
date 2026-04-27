# WRDS Stock Performance Analyser (ACC102 Track 4)

## 1. Problem & User
This project develops an interactive Python-based stock analysis tool that compares two selected stocks with a market benchmark over a chosen period. The intended users are beginner investors, finance/accounting students, and more advanced investors who need both plain-language interpretation and objective risk-adjusted evaluation.

## 2. Data
The project uses daily stock data from the **WRDS CRSP** database.  
**Access date:** April 25, 2026  
**Key fields used:** `permno`, `date`, `prc`, and `cfacpr`.

## 3. Methods (main Python steps)
1. Connect to WRDS and retrieve CRSP security identifiers (`permno`) from ticker symbols.  
2. Query daily stock price data for two stocks and one market benchmark from `crsp.dsf`.  
3. Clean and prepare the data by converting dates, correcting CRSP price sign conventions, and computing adjusted prices.  
4. Calculate descriptive and risk-adjusted indicators, including daily returns, cumulative returns, annual return, annual volatility, Sharpe ratio, maximum drawdown, beta, alpha, tracking error, correlation with the benchmark, and positive-day ratio.  
5. Visualise the results through an interactive Streamlit interface using price charts, cumulative return charts, drawdown charts, rolling volatility charts, and risk-return comparison.  
6. Provide a final academic evaluation section that translates the quantitative results into clear, objective conclusions for both non-specialist and more advanced investors.

## 4. Key Findings
- The tool allows users to compare two stocks and one benchmark within the same selected period using a single interactive interface.
- It transforms raw CRSP price data into interpretable return, risk, and benchmark-relative performance indicators.
- It combines beginner-friendly interpretation with more advanced investment evaluation using multiple metrics rather than relying on return alone.
- It extends the analysis beyond basic performance measures by including maximum drawdown, rolling volatility, beta, alpha, and tracking error.
- It demonstrates how Python analysis can be converted into a small but useful data product rather than being presented only as code output.

## 5. How to run
1. Install the required packages:
   ```bash
   pip install -r requirements.txt
   ```
2. Run the Streamlit app:
   ```bash
   streamlit run app.py
   ```
3. Enter valid WRDS login details and select the stock tickers, benchmark ticker, date range, and evaluation settings.

## 6. Product link / Demo
- **GitHub repository link:** https://github.com/BiteQiu24/ACC102-Track4-WRDS-Stock-Analyser-app.git
- **Interactive app link (if deployed):** optional

## 7. Limitations & next steps
**Limitations**
- The project depends on valid WRDS access, so it cannot run fully without user credentials.
- The analysis is based on a limited set of CRSP price variables and does not incorporate firm fundamentals, macroeconomic variables, or portfolio optimisation.
- Alpha and beta are estimated using a simple benchmark-based framework and should be interpreted as indicative rather than as a full asset-pricing model.
- Results are sensitive to the selected time period and benchmark.

**Next steps**
- Add more models such as downside deviation, Sortino ratio, Treynor ratio, or multi-factor analysis.
- Extend the tool to support more than two stocks and more flexible benchmark selection.
- Improve export functions and interface design for better portfolio-style reporting.
- Add annotation features so users can identify key changes in volatility, drawdown, and benchmark-relative performance.

## Quick Local Run
After downloading or cloning the repository, open a terminal in the repository folder and run:

```bash
pip install -r requirements.txt
streamlit run app.py
