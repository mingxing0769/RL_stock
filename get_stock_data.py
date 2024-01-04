import baostock as bs

OUTPUT = './stock_data'
Date_start = '1990-01-01'
Date_end = '2023-11-13'


class Downloader(object):
    def __init__(self):
        self.date_start = Date_start
        self.date_end = Date_end
        self.adjustflag = "2"  # adjustflag：复权类型，默认不复权：3；1：后复权；2：前复权。已支持分钟线、日线、周线、月线前后复权。
        self.frequency = "d"
        self.output_dir = OUTPUT
        self.fields = "date,code,open,high,low,close,volume,amount,turn,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM," \
                      "tradestatus,isST,adjustflag"

    def exit(self):
        bs.logout()

    def get_codes_by_date(self):
        date = Date_end
        print(date)
        stock_rs = bs.query_all_stock(date)
        stock_df = stock_rs.get_data()
        print(stock_df)
        return stock_df

    def run(self):
        lg = bs.login()
        print('login respond error_code:' + lg.error_code)
        print('login respond  error_msg:' + lg.error_msg)
        stock_df = self.get_codes_by_date()
        for index, row in stock_df.iterrows():
            print(f'processing {row["code"]}')
            df_code = bs.query_history_k_data_plus(row["code"],
                                                   self.fields,
                                                   start_date=self.date_start,
                                                   end_date=self.date_end,
                                                   frequency=self.frequency,
                                                   adjustflag=self.adjustflag).get_data()
            df_code.to_csv(f'{self.output_dir}/{row["code"]}.csv', index=False)
        self.exit()


if __name__ == '__main__':
    downloader = Downloader()
    downloader.run()
