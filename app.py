import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import requests
from langchain.agents import initialize_agent, AgentType
from langchain_community.tools import Tool
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain.memory import ConversationBufferMemory
import matplotlib.pyplot as plt
import seaborn as sns
from langchain_openai import ChatOpenAI
from langchain_core.prompts import PromptTemplate
from datetime import datetime, timedelta

st.set_page_config(page_title="💹 RebalanceAI", layout="wide")
st.title("💹 RebalanceAI – AI-Powered Portfolio Analyzer")

# Access keys from Streamlit secrets
try:
    OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
    NEWS_API_KEY = st.secrets["NEWS_API_KEY"]
    FMP_API_KEY = st.secrets.get("FMP_API_KEY", "")
    GOOGLE_API_KEY = st.secrets.get("GOOGLE_API_KEY", "")
except Exception as e:
    st.error(f"⚠️ Error loading secrets: {e}")
    st.info("Create .streamlit/secrets.toml file with your API keys")
    st.stop()

RF, DAYS_PER_YEAR = 0.035, 252

def get_stock_data(ticker, period="6mo"):
    """Fetch stock data and handle MultiIndex columns"""
    df = yf.download(ticker, period=period, progress=False, auto_adjust=True)
    if df.empty:
        raise ValueError(f"No data returned for {ticker}")
    
    # Handle MultiIndex columns (when yfinance returns ticker names in columns)
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.droplevel(1)
    
    # Ensure we have the Close column
    if "Close" not in df.columns:
        raise ValueError(f"No 'Close' price data available for {ticker}")
    
    # Calculate returns
    returns = df["Close"].pct_change()
    df = df.copy()  # Avoid SettingWithCopyWarning
    df["Return"] = returns
    df.dropna(inplace=True)
    
    return df

def compute_metrics(df):
    """Compute financial metrics from stock data"""
    if len(df) < 2:
        raise ValueError("Insufficient data to compute metrics")
    
    r = df["Return"]
    ann_ret = float(r.mean() * DAYS_PER_YEAR)
    vol = float(r.std() * np.sqrt(DAYS_PER_YEAR))
    sharpe = float((ann_ret - RF) / vol if vol else 0)
    downside = float(r[r < 0].std() * np.sqrt(DAYS_PER_YEAR))
    sortino = float((ann_ret - RF) / downside if downside else 0)
    mdd = float(((df["Close"].cummax() - df["Close"]) / df["Close"].cummax()).max())
    cagr = float((df["Close"].iloc[-1] / df["Close"].iloc[0]) ** (DAYS_PER_YEAR / len(df)) - 1)
    
    return {
        "AnnualReturn": ann_ret,
        "CAGR": cagr,
        "Volatility": vol,
        "Sharpe": sharpe,
        "Sortino": sortino,
        "MaxDrawdown": mdd
    }

def fetch_news(ticker, n=5):
    """Fetch news headlines for a ticker"""
    if not NEWS_API_KEY:
        return ["NewsAPI key missing"]
    
    # Company name mappings
    company_names = {
        "AAPL": "Apple",
        "MSFT": "Microsoft", 
        "NVDA": "NVIDIA",
        "TSLA": "Tesla",
        "JNJ": "Johnson & Johnson",
        "AMZN": "Amazon",
        "VZ": "Verizon"
    }
    
    search_term = company_names.get(ticker, ticker)
    
    # NewsAPI free tier requires date to be within last 30 days
    from_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
    to_date = datetime.now().strftime('%Y-%m-%d')
    
    # Use 'everything' endpoint with date constraints
    url = f"https://newsapi.org/v2/everything"
    params = {
        'q': search_term,
        'language': 'en',
        'sortBy': 'publishedAt',
        'from': from_date,
        'to': to_date,
        'pageSize': n,
        'apiKey': NEWS_API_KEY
    }
    
    try:
        response = requests.get(url, params=params, timeout=10)
        data = response.json()
        
        # Debug: Show API response status
        if data.get("status") == "error":
            return [f"API Error: {data.get('message', 'Unknown error')}"]
        
        if data.get("status") == "ok" and data.get("articles"):
            headlines = [article["title"] for article in data["articles"][:n]]
            if headlines:
                return headlines
            else:
                return [f"No recent news found for {search_term}"]
        else:
            return [f"No articles returned for {search_term}"]
            
    except requests.exceptions.Timeout:
        return ["Request timed out"]
    except Exception as e:
        return [f"Error fetching news: {str(e)}"]

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.3, api_key=OPENAI_API_KEY)

sent_prompt = PromptTemplate.from_template("""
You are a financial sentiment analyst.
Analyze these headlines for {ticker} and classify the overall tone as Positive, Neutral, or Negative with brief justification.

Headlines:
{headlines}

Respond in format: "Sentiment: [Positive/Neutral/Negative]. Justification: [brief reason]"
""")

def analyze_sentiment(ticker):
    """Analyze sentiment from news headlines"""
    headlines = fetch_news(ticker)
    text = "\n".join(headlines)
    
    # Check if we got actual headlines
    if any(err in text.lower() for err in ['error', 'no news', 'no articles', 'missing']):
        return f"Sentiment: Neutral. Justification: {headlines[0]}"
    
    prompt = sent_prompt.format(ticker=ticker, headlines=text)
    try:
        response = llm.invoke(prompt).content
        return response
    except Exception as e:
        return f"Sentiment: Neutral. Error analyzing sentiment: {e}"

def fusion_score(metrics, sentiment):
    """Calculate fusion score from metrics and sentiment"""
    s = sentiment.lower()
    s_factor = 1 if "positive" in s else -1 if "negative" in s else 0
    return float(metrics["Sharpe"] + 0.2 * s_factor)

# Tool wrapper functions for the agent
def get_stock_data_tool(input_str):
    """Tool wrapper for get_stock_data"""
    try:
        parts = input_str.split()
        ticker = parts[0] if parts else input_str
        return str(get_stock_data(ticker))
    except Exception as e:
        return f"Error: {str(e)}"

def compute_metrics_tool(input_str):
    """Tool wrapper for compute_metrics"""
    try:
        parts = input_str.split()
        ticker = parts[0] if parts else input_str
        df = get_stock_data(ticker)
        metrics = compute_metrics(df)
        return str(metrics)
    except Exception as e:
        return f"Error: {str(e)}"

def analyze_sentiment_tool(input_str):
    """Tool wrapper for analyze_sentiment"""
    try:
        parts = input_str.split()
        ticker = parts[0] if parts else input_str
        return analyze_sentiment(ticker)
    except Exception as e:
        return f"Error: {str(e)}"

tickers_input = st.text_input("Enter tickers (comma-separated):", "AAPL, MSFT, NVDA, TSLA, JNJ, AMZN, VZ")
period = st.selectbox("Period", ["3mo", "6mo", "1y"], index=1)

if st.button("🔍 Analyze Portfolio"):
    tickers = [t.strip().upper() for t in tickers_input.split(",")]
    results = []
    progress = st.progress(0)
    
    # Add debug expander
    with st.expander("🔍 Debug: News Fetching", expanded=False):
        test_ticker = tickers[0] if tickers else "AAPL"
        st.write(f"Testing news fetch for {test_ticker}...")
        test_headlines = fetch_news(test_ticker)
        st.write("Headlines received:")
        for i, headline in enumerate(test_headlines, 1):
            st.write(f"{i}. {headline}")
    
    for i, t in enumerate(tickers):
        try:
            df = get_stock_data(t, period)
            m = compute_metrics(df)
            s = analyze_sentiment(t)
            f = fusion_score(m, s)
            
            # Ensure all values are proper Python types
            result = {
                "Ticker": t,
                "AnnualReturn": m["AnnualReturn"],
                "CAGR": m["CAGR"],
                "Volatility": m["Volatility"],
                "Sharpe": m["Sharpe"],
                "Sortino": m["Sortino"],
                "MaxDrawdown": m["MaxDrawdown"],
                "Sentiment": s,
                "FusionScore": f
            }
            results.append(result)
        except Exception as e:
            st.warning(f"Error processing {t}: {str(e)}")
        progress.progress((i + 1) / len(tickers))
    
    if results:
        fusion_df = pd.DataFrame(results).sort_values("FusionScore", ascending=False)
        
        # Calculate weights (handle negative scores)
        min_score = fusion_df["FusionScore"].min()
        adjusted_scores = fusion_df["FusionScore"] - min_score + 0.01
        fusion_df["Weight %"] = (adjusted_scores / adjusted_scores.sum() * 100).round(2)
        
        # Create display dataframe with formatted values
        display_df = fusion_df.copy()
        
        # Format numeric columns safely
        def safe_format(x, decimals=4):
            try:
                if isinstance(x, (int, float, np.number)):
                    return f"{float(x):.{decimals}f}"
                return str(x)
            except:
                return str(x)
        
        display_df["AnnualReturn"] = display_df["AnnualReturn"].apply(lambda x: safe_format(x, 4))
        display_df["CAGR"] = display_df["CAGR"].apply(lambda x: safe_format(x, 4))
        display_df["Volatility"] = display_df["Volatility"].apply(lambda x: safe_format(x, 4))
        display_df["Sharpe"] = display_df["Sharpe"].apply(lambda x: safe_format(x, 4))
        display_df["Sortino"] = display_df["Sortino"].apply(lambda x: safe_format(x, 4))
        display_df["MaxDrawdown"] = display_df["MaxDrawdown"].apply(lambda x: safe_format(x, 6))
        display_df["FusionScore"] = display_df["FusionScore"].apply(lambda x: safe_format(x, 4))
        
        st.dataframe(display_df, use_container_width=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("📊 Sharpe Ratios")
            fig, ax = plt.subplots(figsize=(8, 4))
            sns.barplot(x=fusion_df["Ticker"], y=fusion_df["Sharpe"], ax=ax, palette="viridis")
            ax.set_ylabel("Sharpe Ratio")
            ax.set_xlabel("Ticker")
            plt.xticks(rotation=45)
            st.pyplot(fig)
        
        with col2:
            st.subheader("🥧 Allocation by Fusion Score")
            fig2, ax2 = plt.subplots(figsize=(8, 4))
            weights = fusion_df["Weight %"].values
            ax2.pie(weights, labels=fusion_df["Ticker"], autopct="%1.1f%%", startangle=90)
            ax2.axis('equal')
            st.pyplot(fig2)
        
        # Generate report
        report_prompt = PromptTemplate.from_template("""
        Write a concise investment analysis for this portfolio (3-4 paragraphs).
        Include: market overview, quantitative highlights, sentiment summary, allocation reasoning, and risk disclaimer.

        Portfolio Data:
        {table}
        """)
        report_text = llm.invoke(report_prompt.format(table=fusion_df.to_string(index=False))).content
        st.markdown("## 📄 Investment Report")
        st.markdown(report_text)
    else:
        st.error("No data could be retrieved. Please check your tickers.")

st.markdown("---")
st.header("💬 Investment Chat")
query = st.text_input("Ask a question (e.g., 'Should I invest in NVDA?')")
if st.button("Ask"):
    if query:
        tools = [
            Tool(
                name="GetStockData",
                func=get_stock_data_tool,
                description="Fetch stock price data for a ticker symbol. Input should be a ticker symbol like 'AAPL' or 'NVDA'."
            ),
            Tool(
                name="ComputeMetrics",
                func=compute_metrics_tool,
                description="Compute financial metrics (returns, volatility, Sharpe ratio) for a ticker. Input should be a ticker symbol."
            ),
            Tool(
                name="AnalyzeSentiment",
                func=analyze_sentiment_tool,
                description="Analyze news sentiment for a stock ticker. Input should be a ticker symbol."
            )
        ]
        memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
        agent = initialize_agent(
            tools, 
            llm, 
            agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
            verbose=True, 
            memory=memory, 
            handle_parsing_errors=True
        )
        try:
            with st.spinner("Thinking..."):
                result = agent.invoke({"input": query})
                st.success(result["output"])
        except Exception as e:
            st.error(f"Error: {str(e)}")
    else:
        st.warning("Please enter a question.")
