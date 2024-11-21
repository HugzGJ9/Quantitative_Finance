from ib_insync import *
from datetime import datetime

# Initialize the IB instance
ib = IB()
ib.connect('127.0.0.1', 4002, clientId=1)

def request_option_chain(ib, symbol='NG', exchange='NYMEX'):
    try:
        # Get all future contracts for NG
        future_contracts = ib.reqContractDetails(Future(symbol=symbol, exchange=exchange))
        if not future_contracts:
            print(f"No future contracts found for {symbol} on {exchange}")
            return

        # Sort the future contracts by expiration date
        future_contracts = sorted(
            future_contracts,
            key=lambda cd: datetime.strptime(cd.contract.lastTradeDateOrContractMonth, '%Y%m%d')
        )

        # Use the earliest future contract
        future = future_contracts[0].contract
        print(f"Using future contract: {future.localSymbol}, Expiry: {future.lastTradeDateOrContractMonth}")

        # Get option chain for this future
        option_chains = ib.reqSecDefOptParams(
            future.symbol, future.exchange, future.secType, future.conId)
        # Pick the option chain with the matching exchange
        chain = next((c for c in option_chains if c.exchange == exchange), None)
        if chain is None:
            print(f"No option chain found for {symbol} on {exchange}")
            return

        # Print chain details
        print(f"Exchange: {chain.exchange}")
        print(f"Underlying ConId: {chain.underlyingConId}")
        print(f"Trading Class: {chain.tradingClass}")
        print(f"Multiplier: {chain.multiplier}")
        print(f"Expirations: {chain.expirations}")
        print(f"Strikes: {chain.strikes}")
        print('-' * 50)

        # Select the earliest expiration and desired strikes
        expirations = sorted(
            chain.expirations,
            key=lambda x: datetime.strptime(x, '%Y%m%d')
        )
        expiration = expirations[0]  # Earliest expiration

        strikes = [2.6]  # Desired strike(s)

        option_contracts = []
        for strike in strikes:
            option = Option(
                symbol=future.symbol,  # Use the future's symbol
                secType=future.secType,         # Important for options on futures
                lastTradeDateOrContractMonth=expiration,
                strike=strike,
                right='C',  # 'C' for Call, 'P' for Put
                exchange=exchange,
                currency='USD'
            )
            option_contracts.append(option)

        # Qualify contracts to get full details
        qualified_contracts = ib.qualifyContracts(*option_contracts)
        if not qualified_contracts:
            print(f"Option contract could not be qualified.")
            return

        # Request market data for these option contracts
        for option in qualified_contracts:
            ticker = ib.reqMktData(option, '', False, False)
            ib.sleep(2)  # Wait for data to be received
            bid = ticker.bid if ticker.bid is not None else 'N/A'
            ask = ticker.ask if ticker.ask is not None else 'N/A'
            iv = ticker.modelGreeks.impliedVol if ticker.modelGreeks else 'N/A'
            print(f"Option: {option.localSymbol}, Bid: {bid}, Ask: {ask}, Implied Volatility: {iv}")
    except Exception as e:
        print(f"Failed to get option chain response: {e}")

if __name__ == '__main__':
    request_option_chain(ib, symbol='NG', exchange='NYMEX')
    symbol = 'NG'
    strike = 2.0
    expiry_option = '20241029'
    expiry = '202411'
    exchange = 'NYMEX'
    request_option_chain()
    call_option = Option(symbol=symbol, lastTradeDateOrContractMonth=expiry, strike=strike, right='C', exchange=exchange, currency='USD')
    underlying = Future(symbol=symbol, exchange=exchange, lastTradeDateOrContractMonth=expiry)
    ib.qualifyContracts(call_option, underlying)
    ib.disconnect()
