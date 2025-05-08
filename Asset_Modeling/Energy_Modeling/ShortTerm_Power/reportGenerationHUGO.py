from API.SUPABASE.save import saveDailyGenerationTS
from Asset_Modeling.Energy_Modeling.ShortTerm_Power.buildGenerationReport import buildMonthlyTable, buildGenerationForecastEmail, fetchGenerationReport
from Asset_Modeling.Energy_Modeling.data.data import fetchRESGenerationMonthlyData
from Model.Power.RESPowerGeneration_forecast import getGenerationForecastReport

generation_forecast = getGenerationForecastReport(model_name="model_RES_generation_LGBMR_cleaned_pt")

res_generation_month, res_generation_day = fetchRESGenerationMonthlyData("FR")
monthly_stats = buildMonthlyTable(res_generation_month, res_generation_day)
saveDailyGenerationTS(generation_forecast, 'HUGO')

tables, image_cid, img_data = fetchGenerationReport(generation_forecast, monthly_stats)
title = f'Hugo RES Generation Forecast FR {generation_forecast.index[0].strftime("%Y-%m-%d")}'

buildGenerationForecastEmail(title=title, tables=tables, image_cid=image_cid, img_data=img_data)