import pandas as pd

def convert_data(path = "padavine-bjelasnica.ods", prefix="P"):

	name=path.split("-")[1].split(".")[0]
        
	df = pd.read_excel(path, engine="odf")

	df = df.rename(columns={name.upper(): "year"})

	month_map = {  
    		"I": 1, "II": 2, "III": 3, "IV": 4,
    		"V": 5, "VI": 6, "VII": 7, "VIII": 8,
    		"IX": 9, "X": 10, "XI": 11, "XII": 12
	}

	df_long = df.melt(
    		id_vars=["year"],
    		value_vars=month_map.keys(),
    		var_name="month",
    		value_name=f"{prefix}_{name}"
	)

	df_long["month_num"] = df_long["month"].map(month_map)
	df_long["date"] = pd.to_datetime(
    		dict(year=df_long["year"], month=df_long["month_num"], day=1)
	)

	df_final = (
    		df_long[["date", f"{prefix}_{name}"]]
    		.sort_values("date")
    		.reset_index(drop=True)
	)

	print(df_final)
	return df_final
