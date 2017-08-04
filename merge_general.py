def merge_general(df1, df2, how='inner', on_cols=None):

	# left merge. 
	on_cols = df1.columns if not on_cols else on_cols
	# Use a tri structure instead where you move down the leaves as you find matching elements. 
	merged = {}
	i = 0
	for row1 in df1.iterrows():
		max_overlap = 0
		for row2 in df2.iterrows():
			# Case when one of the elements is different and not missing: don't update the merged table
			# count the number of fields that are the same
			print(row1, row2)
			overlap = sum([row1[col] == row2[col] for col in on_cols])
			if overlap > max_overlap:
				row2['overlap'] = overlap
				merged[i] = row2
		i += 1


merge_general(vin, epa, how='left')