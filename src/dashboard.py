import streamlit as st
import pandas as pd
import os
import sys

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.append(BASE_DIR)

from deltalake import DeltaTable
import plotly.express as px
import plotly.graph_objects as go
import torch
import numpy as np
from src.models.training import WeatherLSTM

st.set_page_config(page_title="WeatherReport Dashboard", layout="wide")
SILVER_DIR = os.path.join(BASE_DIR, "data", "silver")
MODEL_PATH = os.path.join(BASE_DIR, "src", "models", "lstm_weather.pth")

@st.cache_resource
def load_model():
    if not os.path.exists(MODEL_PATH):
        return None, None
    checkpoint = torch.load(MODEL_PATH, map_location=torch.device('cpu'), weights_only=False)
    model = WeatherLSTM(checkpoint['input_size'], checkpoint['hidden_size'], checkpoint['num_layers'], checkpoint['output_size'])
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    return model, checkpoint

def predict_next_day(df, model, checkpoint, reference_date):

    scaler = checkpoint['scaler']
    feature_cols = checkpoint['feature_cols']
    df_upto = df[df['Data Medicao'] <= reference_date].sort_values('Data Medicao')
    last_6_days = df_upto.tail(6).copy()
    if len(last_6_days) < 6:
        return None
    input_data = scaler.transform(last_6_days[feature_cols])
    input_tensor = torch.tensor(input_data, dtype=torch.float32).unsqueeze(0)
    with torch.no_grad():
        prediction_scaled = model(input_tensor).numpy()
    prediction_real = scaler.inverse_transform(prediction_scaled)
    pred_df = pd.DataFrame(prediction_real, columns=feature_cols)
    next_date = reference_date + pd.Timedelta(days=1)
    pred_df['Data Medicao'] = next_date
    return pred_df

def main():
    st.title("🌤️ WeatherReport: Intelligence Dashboard")
    st.markdown("Monitoramento de dados via **Delta Lake** e análise de performance **LSTM**.")
    if not os.path.exists(SILVER_DIR) or not os.listdir(SILVER_DIR):
        st.warning("Tabela Delta Silver não encontrada em `data/silver`. Execute o pipeline primeiro.")
        return
    tab_monitor, tab_results = st.tabs(["📊 Monitoramento", "🧠 LSTM Results"])
    try:
        dt = DeltaTable(SILVER_DIR)
        df_full = dt.to_pandas()
        if 'Data Medicao' in df_full.columns:
            df_full['Data Medicao'] = pd.to_datetime(df_full['Data Medicao'])
            df_full = df_full.sort_values('Data Medicao')
        model, checkpoint = load_model()
        with tab_monitor:
            st.sidebar.header("Configurações")
            st.sidebar.info(f"Versão da Tabela Delta: {dt.version()}")
            min_date = df_full['Data Medicao'].min().date()
            max_date = df_full['Data Medicao'].max().date()
            st.sidebar.subheader("Seleção Temporal")
            selected_date = st.sidebar.date_input("Data de Referência", value=max_date, min_value=min_date, max_value=max_date)
            ref_date = pd.Timestamp(selected_date)
            if st.sidebar.button("🔄 Atualizar Dados"):
                st.rerun()
            df = df_full[df_full['Data Medicao'] <= ref_date].copy()
            pred_df = None
            real_next_day = pd.DataFrame()
            if model and len(df) >= 6:
                st.sidebar.success("Modelo LSTM Carregado")
                pred_df = predict_next_day(df_full, model, checkpoint, ref_date)
                if pred_df is not None:
                    next_day = pred_df['Data Medicao'].iloc[0]
                    st.subheader(f"🔮 Predição para: {next_day.strftime('%d/%m/%Y')}")
                    real_next_day = df_full[df_full['Data Medicao'].dt.date == next_day.date()]
                    p_cols = st.columns(4)
                    cols_to_show = checkpoint['feature_cols'][:4]
                    for i, col in enumerate(cols_to_show):
                        val_pred = pred_df[col].iloc[0]
                        if not real_next_day.empty:
                            val_real = real_next_day[col].iloc[0]
                            delta = val_pred - val_real
                            p_cols[i % 4].metric(col, f"Prev: {val_pred:.2f}", f"Real: {val_real:.2f}", delta_color="off")
                        else:
                            last_val = df[col].iloc[-1]
                            delta = val_pred - last_val
                            p_cols[i % 4].metric(col, f"{val_pred:.2f}", f"Δ: {delta:.2f}")
            else:
                st.sidebar.warning("Modelo LSTM não encontrado ou dados insuficientes.")
            st.sidebar.subheader("Filtros de Variáveis")
            all_cols = [col for col in df_full.columns if col not in ['Data Medicao', 'injection_timestamp']]
            selected_cols = st.sidebar.multiselect("Variáveis para exibir", all_cols, default=all_cols[:3])
            st.subheader(f"📊 Histórico Silver (até {selected_date.strftime('%d/%m/%Y')})")
            col1, col2, col3 = st.columns(3)
            with col1: st.metric("Primeira Medição", df_full['Data Medicao'].min().strftime('%d/%m/%Y'))
            with col2: st.metric("Data Selecionada", selected_date.strftime('%d/%m/%Y'))
            with col3: st.metric("Registros Visíveis", len(df))
            st.divider()
            if selected_cols:
                st.subheader("Séries Temporais (Histórico + Predição)")
                df_viz = pd.concat([df.tail(20), pred_df], ignore_index=True) if pred_df is not None else df.tail(20)
                fig = px.line(df_viz, x='Data Medicao', y=selected_cols, title="Visualização Temporal", labels={"value": "Valor", "variable": "Variável", "Data Medicao": "Data"})
                if pred_df is not None:
                    for col in selected_cols:
                        fig.add_scatter(x=pred_df['Data Medicao'], y=pred_df[col], mode='markers', marker=dict(size=10), name=f'Prev: {col}')
                fig.update_layout(hovermode="x unified", legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
                st.plotly_chart(fig, use_container_width=True)
                st.subheader("Detalhamento por Variável")
                cols_per_row = 2
                for i in range(0, len(selected_cols), cols_per_row):
                    row_cols = st.columns(cols_per_row)
                    for j in range(cols_per_row):
                        if i + j < len(selected_cols):
                            col_name = selected_cols[i+j]
                            with row_cols[j]:
                                fig_sep = px.area(df_viz, x='Data Medicao', y=col_name, title=col_name)
                                st.plotly_chart(fig_sep, use_container_width=True)
            st.divider()
            is_future = real_next_day.empty
            st.subheader(f"📋 Comparativo de Predição ({'Futuro' if is_future else 'Validado'})")
            if pred_df is not None:
                p_df = pred_df.head(1).drop(columns=['Data Medicao'], errors='ignore')
                display_pred = p_df.T.reset_index()
                display_pred.columns = ['Variável', 'Valor Previsto']
                if not is_future:
                    r_df = real_next_day.head(1).drop(columns=['Data Medicao', 'injection_timestamp'], errors='ignore')
                    real_vals = r_df.T.reset_index()
                    if real_vals.shape[1] == 2:
                        header_real = f"Valor Real ({next_day.strftime('%d/%m/%Y')})"
                        real_vals.columns = ['Variável', header_real]
                        display_pred = display_pred.merge(real_vals, on='Variável')
                        display_pred['Erro Absoluto'] = (display_pred['Valor Previsto'] - display_pred[header_real]).abs()
                    else: st.error(f"Erro na estrutura dos dados reais")
                st.dataframe(display_pred, use_container_width=True)
            else: st.info("Predição não disponível")
            with st.expander("Ver Dados Brutos (Tabela)"): st.dataframe(df, use_container_width=True)
            with st.expander("Histórico de Versões Delta"):
                history = dt.history()
                st.table(history)
        with tab_results:
            st.header("📈 Resultados do Último Treinamento (Conjunto de Teste)")
            if checkpoint and 'y_test_real' in checkpoint and 'y_pred_real' in checkpoint:
                feature_cols = checkpoint['feature_cols']
                y_test = checkpoint['y_test_real']
                y_pred = checkpoint['y_pred_real']
                num_samples = len(y_test)
                st.info(f"Exibindo performance para {num_samples} amostras do conjunto de teste.")
                for i, col in enumerate(feature_cols):
                    st.subheader(f"Variável: {col}")
                    res_df = pd.DataFrame({ 'Amostra': np.arange(num_samples), 'Real': y_test[:, i], 'Predição': y_pred[:, i] })
                    fig_res = px.line(res_df, x='Amostra', y=['Real', 'Predição'], title=f"Real vs Predição: {col}")
                    st.plotly_chart(fig_res, use_container_width=True)
            else: st.warning("Dados de teste não encontrados no modelo. Execute um novo treinamento primeiro.")
    except Exception as e:
        st.error(f"Erro no Dashboard: {e}")
        import traceback
        st.code(traceback.format_exc())

if __name__ == "__main__":
    main()
