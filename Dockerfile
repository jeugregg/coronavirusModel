FROM python:3.7-slim
WORKDIR /app
COPY requirements_light.txt /app/
COPY app.py /app/
COPY data_meteo_fr.json /app/ 
COPY df_feat_fr.csv /app/
COPY df_plot_pred.csv /app/
COPY df_plot_pred_all.csv /app/
COPY df_dep_r0.csv /app/
COPY pt_fr_test_last.csv /app/
COPY sources/departements-avec-outre-mer_simple.json /app/sources/
COPY settings.py /app/
COPY my_helpers/data_maps.py /app/my_helpers/
COPY my_helpers/data_plots.py /app/my_helpers/
COPY my_helpers/dates.py /app/my_helpers/
COPY my_helpers/meteo.py /app/my_helpers/
COPY my_helpers/model.py /app/my_helpers/
RUN pip install -r requirements_light.txt
EXPOSE 80
CMD ["python", "app.py"]