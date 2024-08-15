# Natural-Language-to-MYSql-Query
<h1>Interactive NL to SQL Query Generator</h1>
<p>An interactive NL to SQL Query Generator powered by BERT and Gemini LLM, enabling seamless database querying through natural language inputs. Deployed with Streamlit for an intuitive user experience.</p>

<h2>Features</h2>
<ul>
    <li>Natural language input to SQL query conversion.</li>
    <li>Uses BERT for text vectorization and similarity matching.</li>
    <li>Gemini LLM for column name replacement and SQL query generation.</li>
    <li>Handles multiple matches using LIKE operator with wildcard characters.</li>
    <li>Interactive interface deployed with Streamlit.</li>
</ul>

<h2>Technologies Used</h2>
<ul>
    <li>Python</li>
    <li>BERT (Hugging Face Transformers)</li>
    <li>Google Gemini LLM API</li>
    <li>MySQL</li>
    <li>Pandas</li>
    <li>Streamlit</li>
</ul>

<h2>Getting Started</h2>
<ol>
    <li>Clone the repository.</li>
    <li>Install the required packages from <code>requirements.txt</code>.</li>
    <li>Set up your MySQL database with the required schema.</li>
    <li>Replace the <code>GEMINI_API_KEY</code> in the code with your actual Gemini API key.</li>
    <li>Run the Streamlit app using the command <code>streamlit run app.py</code>.</li>
</ol>

<h2>Future Enhancements</h2>
<ul>
    <li>Improved accuracy of record matching using advanced ML models.</li> 
    <li>Expands support for more complex SQL queries.</li>
    <li>Integrate additional language models for enhanced query interpretation.</li>
</ul>

<h2>Examples</h2>

![Screenshot_14-8-2024_234823_localhost](https://github.com/user-attachments/assets/3c9bd628-703a-4d8f-aa4f-417206469945)
<br><br>
![Screenshot_14-8-2024_234934_localhost](https://github.com/user-attachments/assets/30b08921-9c69-44e6-b786-5a11605dc800)
<br><br>
![Screenshot_14-8-2024_235229_localhost](https://github.com/user-attachments/assets/70808384-a9b8-430a-aadb-9f7acbeea4ed)
