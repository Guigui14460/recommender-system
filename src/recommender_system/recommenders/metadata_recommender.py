# meta['director'] = meta['director'].apply(lambda x: [x,x,x])
# s = meta.apply(lambda x: pd.Series(x['keywords'], dtype="object"),axis=1).stack().reset_index(level=1, drop=True)
# s.name = 'keyword'
# s = s[s > 1]
