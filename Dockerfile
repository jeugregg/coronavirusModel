FROM python:3.7-slim
WORKDIR /app
COPY requirements_light.txt /app/
COPY app.py /app/
COPY data_meteo_fr.json /app/ 
COPY df_feat_fr.csv /app/
COPY mdl_multi_step_pos_fr /app/mdl_multi_step_pos_fr/
COPY df_plot_pred.csv /app/
COPY df_plot_pred_all.csv /app/
COPY sources/departements-avec-outre-mer_simple.json /app/sources/
RUN pip install -r requirements_light.txt
EXPOSE 80
CMD ["python", "app.py"]