import quantopian.algorithm as algo
import pandas as pd
import quandl
from quantopian.pipeline import Pipeline
from quantopian.pipeline.data.builtin import USEquityPricing
from quantopian.pipeline.filters import QTradableStocksUS
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC, NuSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn import preprocessing
from collections import Counter
from datetime import timedelta


def initialize(context):
    """
    Called once at the start of the algorithm.
    """
    
    auth_code ='C_g4_bPDdjFrsdaf8P_K'
    database ='ZES'
    
    #Securities we want to trade
    dataset ='AAPL'
    
    url = 'https://quandl.com/api/v1/datasets/{0}/{1}.csv?auth_code={2}'
    url = url.format(database, dataset,auth_code)
    algo.fetch_csv(url,  date_column='DATE',  symbol=dataset,  date_format='%Y-%m-%d',  post_func=post_func)
    
    algo.set_symbol_lookup_date('2018-10-21')
    
    ## Trailing stop loss
    context.stop_loss_pct = 0.995
    
    # We will weight each asset equally and leave a 5% cash
    # reserve. - actually this is sort of good idea
    context.weight = 0.95 / len(context.security_list)
    
    context.investment_size = (context.portfolio.cash*context.weight)                           

    context.historical_bars = 100
    context.feature_window = 3
    
    # Rebalance every day, 1 hour after market open.
    algo.schedule_function(
        rebalance,
        algo.date_rules.every_day(),
        algo.time_rules.market_open(hours=1),
    )

    # Record tracking variables at the end of each day.
    algo.schedule_function(
        record_vars,
        algo.date_rules.every_day(),
        algo.time_rules.market_close(),
    )

    # Create our dynamic stock selector.
    algo.attach_pipeline(make_pipeline(), 'pipeline')


def make_pipeline():
    """
    A function to create our dynamic stock selector (pipeline). Documentation
    on pipeline can be found here:
    https://www.quantopian.com/help#pipeline-title
    """

    # Base universe set to the QTradableStocksUS
    base_universe = QTradableStocksUS()

    # Factor of yesterday's close price.
    yesterday_close = USEquityPricing.close.latest

    pipe = Pipeline(
        columns={
            'close': yesterday_close,
        },
        screen=base_universe
    )
    return pipe


def before_trading_start(context, data):
    """
    Called every day before market open.
    """
    context.output = algo.pipeline_output('pipeline')

    # These are the securities that we are interested in trading each day.
    context.security_list = context.output.index


def rebalance(context, data):
    """
    Execute orders according to our schedule_function() timing.
    """
    
    df = quandl.get("ITF/SMA");
    data = post_func(df)
    
    
    price_history = data.history(context.security_list, fields="price", bar_count=100, frequency="1d")
        
    try: 
        # For loop for each stock traded everyday:
        for s in context.security_list:
            
            start_bar = context.feature_window
            price_list = price_history[s].tolist()
            past = data.current(s,'past_data')
            pastlist = parse_list(past)

            X = []
            y = []
    
            bar = start_bar
            
            # Loop for each machine learning data set
            while bar < len(price_list)-1:
   
                try: 
                    end_price = price_list[bar]
                    start_price = price_list[bar-1]
                    
                    features = pastlist[(bar-3)*4: bar*4]
            
                    if end_price > start_price:
                        label = 1
                    else:
                        label = -1

                    bar +=1 
            
                    X.append(features)
                    y.append(label)
         
                except Exception as e:
            
                    bar +=1
                    print(('feature creation', str(e)))
            
            print ('len(X1)',len(X))
            
            # Call the machined learning model
            clf1 = RandomForestClassifier(n_estimators=100)
            clf2 = LinearSVC()
            clf3 = NuSVC()
            clf4 = LogisticRegression()
            
            # Rrepare the attribute information for prediction
            current_features=pastlist[(bar-3)*4: bar*4]
            
            X.append(current_features)
            print ('len(X2)',len(X))
            
            X = preprocessing.scale(X)
    
            current_features = X[-1:]
            X = X[:-1]
            
            #print current_features
            print ('len(X)',len(X))
            print ('len(y)',len(y))
            
            # Build the model
            clf1.fit(X,y)
            clf2.fit(X,y)
            clf3.fit(X,y)
            clf4.fit(X,y)
    
            # Predict the results 
            p1 = clf1.predict(current_features)[0]
            p2 = clf2.predict(current_features)[0]
            p3 = clf3.predict(current_features)[0]
            p4 = clf4.predict(current_features)[0]
     
            # If 3 out of 4 prediction votes for one same results, this results will be promted to be the one I will use. 
            if Counter([p1,p2,p3,p4]).most_common(1)[0][1] >= 3:
                p = Counter([p1,p2,p3,p4]).most_common(1)[0][0]
        
            else: 
                p = 0
        
            print(('Prediction',p))         
            
            current_price = data.current(s, 'price')
            current_position = context.portfolio.positions[s].amount
            cash = context.portfolio.cash
            
            open_orders = algo.get_open_orders()
            
            # Everyday's trading activities: 
            if (p == 1):
                if s not in open_orders:
                    algo.order_target_percent(s, context.weight, style=algo.StopOrder(context.stop_loss_pct*current_price))
                    cash-=context.investment_size
            elif (p == -1):
                if s not in open_orders:
                    algo.order_target_percent(s,-context.weight)
   
    except Exception as e:
        print(str(e))

def record_vars(context, data):
    """
    Plot variables at the end of each day.
    """
    
    long_count = 0
    short_count = 0

    for position in context.portfolio.positions.itervalues():
        if position.amount > 0:
            long_count += 1
        if position.amount < 0:
            short_count += 1
            
    algo.record(num_long=long_count, num_short=short_count, leverage=context.account.leverage)

def handle_data(context, data):
    """
    Called every minute.
    """
    pass

def parse_list(string_list):
    """
    Parses a string and returns it in list format
    """
    # Remove the '[' and ']'
    string_list = string_list[1:-1].split(',')
    # Convert to float
    string_list = [float(s) for s in string_list]
    return string_list

def fill_func(df, row, num_dates):
    """
        Should be applied to every row of a dataframe. Reaches for the past thirty days of each dataframe,
        appends the data to a string, returns the string which should be unpacked later on.

    """
    # Instantiate variables
    past_data = []
    # The current date is the name of the Series (row) being passed in 
    current_date = row.name
    # print ("current_date ", current_date)
    # Iterate through the number of dates from 0->num_dates
    for i in range(num_dates):
        # How many days to get back, calls get_day_delta for accurate delta assessment
        day_delta = delta(current_date)
        # print ("day delta ", day_delta)
        # Get the current_date and update the current_date to minus day_delta from the date
        # To get the appropriate past date
        current_date = current_date - timedelta(days=day_delta)
        #print ("changed current_date ", current_date)
        try:
            #: Get the price at the given current_date found by get_day_delta
            data = df.iloc[current_date]['sentiment']
            # print ("current date ", current_date, "data " ,data)
            past_data.append(data)
            
            data = df.iloc[current_date]['sentiment high']- df.iloc[current_date]['sentiment low']
            past_data.append(data)
            
            data = df.iloc[current_date]['news volume']
            past_data.append(data)
            
            data = df.iloc[current_date]['news buzz']
            past_data.append(data)
            
            #print ("past data " ,past_data)
        except KeyError:
            #: No data for this date, pass
            pass
    # print str(past_data)
    
    # Return the a list made into a string
    return str(past_data)

def post_func(df):
    """
    Applied to each row a function that searches for previous days sentiments analysis and compresses
    into string for each to be parsed later
    """
    
    df = pd.DataFrame(df)
    df['past_data'] = df.apply(lambda row: fill_func(df, row, 99), axis=1)
   
    return df

def delta(current_date):
    """
        Takes in the current date, checks it's day of week, and returns an appropriate date_delta
        E.g. if it's a Monday, the previous date should be Friday, not Sunday
    """
    if current_date.isoweekday() == 1:
        return 3
    else:
        return 1
