# Modern HTML Template
HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Trading Report - {{ ticker }}</title>
    <style>
        :root {
            --primary-color: #3498db;
            --positive-color: #2ecc71;
            --negative-color: #e74c3c;
            --neutral-color: #95a5a6;
        }

        body { 
            font-family: 'Segoe UI', system-ui; 
            margin: 2rem;
            color: #2c3e50;
        }

        .kpi-grid { 
            display: grid; 
            grid-template-columns: repeat(auto-fill, minmax(220px, 1fr)); 
            gap: 1.5rem;
            margin: 2rem 0;
        }

        .kpi-box { 
            background: white; 
            padding: 1.5rem; 
            border-radius: 10px; 
            text-align: center;
            box-shadow: 0 4px 6px rgba(0, 0, 0, 0.05);
            transition: transform 0.2s;
            border-left: 4px solid {{ color }};
        }

        .kpi-box:hover {
            transform: translateY(-3px);
            box-shadow: 0 6px 8px rgba(0, 0, 0, 0.08);
        }

        .chart-container { 
            margin: 3rem 0; 
            border: 1px solid #e0e0e0; 
            border-radius: 12px; 
            padding: 1.5rem;
            background: white;
        }

        .tabs {
            margin: 2rem 0;
            border-bottom: 2px solid #f0f0f0;
        }

        .tablink {
            background: none;
            border: none;
            padding: 1rem 2rem;
            margin-right: 1rem;
            cursor: pointer;
            font-weight: 500;
            color: var(--neutral-color);
            transition: all 0.2s;
        }

        .tablink.active {
            color: var(--primary-color);
            border-bottom: 3px solid var(--primary-color);
        }

        .insights {
            background: #f8f9fa;
            padding: 1.5rem;
            border-radius: 8px;
            margin: 2rem 0;
        }
    </style>
</head>
<body>
    <h1 style="color: var(--primary-color); border-bottom: 2px solid var(--primary-color); padding-bottom: 0.5rem;">
        Trading Report - {{ ticker }}
    </h1>

    <div class="tabs">
        <button class="tablink active" onclick="openTab(event, 'overview')">Overview</button>
        <button class="tablink" onclick="openTab(event, 'volatility')">Volatility</button>
        <button class="tablink" onclick="openTab(event, 'momentum')">Momentum</button>
    </div>

    <div id="overview" class="tabcontent" style="display: block;">
        <div class="kpi-grid">
            {% for k,v in kpis.items() %}
            <div class="kpi-box" style="border-color: 
                {% if 'Return' in k or 'Ratio' in k %}
                    {{ 'var(--positive-color)' if v > 0 else 'var(--negative-color)' }}
                {% else %} 
                    var(--primary-color)
                {% endif %}">
                <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">
                    {% if 'Return' in k %}📈{% elif 'Volatility' in k %}🌪️{% endif %}
                </div>
                <h3>{{ k }}</h3>
                <p style="font-size: 1.4rem; font-weight: 600;">
                    {{ "%.2f"|format(v) }}{{ '%' if 'Return' in k or 'Volatility' in k else '' }}
                </p>
            </div>
            {% endfor %}
        </div>
        <div class="chart-container">
            <h2>ID Price Analysis</h2>
            {{ figures[7] }}
        </div>
        <div class="chart-container">
            <h2>Price Analysis</h2>
            {{ figures[0] }}
        </div>
        <div class="chart-container">
            <h2>MACD Prior Week</h2>
            {{ figures[3] }}
        </div>
        <div class="chart-container">
            <h2>RSI ID</h2>
            {{ figures[8] }}
        </div>
        <div class="chart-container">
            <h2>RSI</h2>
            {{ figures[2] }}
        </div>
    </div>

    <div id="volatility" class="tabcontent">
        <div class="kpi-grid">
                {% for k,v in kpis_vol.items() %}
                <div class="kpi-box" style="border-color: 
                    {% if 'Return' in k or 'Ratio' in k %}
                        {{ 'var(--positive-color)' if v > 0 else 'var(--negative-color)' }}
                    {% else %} 
                        var(--primary-color)
                    {% endif %}">
                    <div style="font-size: 1.8rem; margin-bottom: 0.5rem;">
                        {% if 'Return' in k %}📈{% elif 'Volatility' in k %}🌪️{% endif %}
                    </div>
                    <h3>{{ k }}</h3>
                    <p style="font-size: 1.4rem; font-weight: 600;">
                        {{ "%.2f"|format(v) }}{{ '%' if 'Return' in k or 'Volatility' in k else '' }}
                    </p>
                </div>
                {% endfor %}
            </div>
        <div class="chart-container">
            <h2>Intraday Volatility Trend</h2>
            {{ figures[6] }}
        </div>
        <div class="chart-container">
            <h2>Daily Volatility Trend</h2>
            {{ figures[5] }}
        </div>
        <div class="chart-container">
            <h2>Weekly Volatility Trend</h2>
            {{ figures[4] }}
        </div>

    </div>

    <script>
        function openTab(evt, tabName) {
            var tabcontent = document.getElementsByClassName("tabcontent");
            for (var i = 0; i < tabcontent.length; i++) {
                tabcontent[i].style.display = "none";
            }

            var tablinks = document.getElementsByClassName("tablink");
            for (var i = 0; i < tablinks.length; i++) {
                tablinks[i].className = tablinks[i].className.replace(" active", "");
            }

            document.getElementById(tabName).style.display = "block";
            evt.currentTarget.className += " active";
        }
    </script>
</body>
</html>
"""