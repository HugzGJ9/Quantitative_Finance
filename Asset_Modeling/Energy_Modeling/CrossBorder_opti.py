import numpy as np

from Logger import Logger

#SPAIN (SP) data
SP_SR_Mc = 5.0
SP_Gas_Mc = 60.0
SP_WON_Mc = 15.0

SP_peak_SR_Gene = 3000.0
SP_peak_Gas_Gene = 5000.0
SP_peak_WON_Gene = 0.0

SP_peak_demand = 2500.0

SP_op_SR_Gene = 0.0
SP_op_Gas_Gene = 5000.0
SP_op_WON_Gene = 1000.0

SP_op_demand = 4000.0

#FRANCE (FR) data
FR_NUC_Mc = 20.0
FR_WON_Mc = 10.0

FR_peak_NUC_Gene = 5000.0
FR_peak_WON_Gene = 500.0

FR_peak_demand = 6000.0

FR_op_NUC_Gene = 5000.0
FR_op_WON_Gene = 2000.0

FR_op_demand = 7000.0

MAX_INJECTION = 2000.0
mylogger = Logger.LOGGER()

def analyse_supply(generation):
    supply = 0.0
    for indice in range(len(generation)-1):
        supply+=generation[indice][0]
    if supply>generation[-1]:
        mylogger.logger.info('OVER SUPPLY LOAD.')
    else:
        mylogger.logger.info('UNDER SUPPLY LOAD.')
    return supply - generation[-1]


def sort_by_price(items):
    def get_price(item):
        if isinstance(item, tuple) and len(item) >= 2:
            # Price is the second element of the tuple
            return item[1]
        elif isinstance(item, (int, float)):
            # Price is the item itself
            return item
        else:
            # Assign a high price if price cannot be determined
            return float('inf')

    # Sort the items using the get_price function as the key
    return sorted(items, key=get_price)
def meritorder(generation):
    demand = generation[-1]
    # Exclude the demand from the generation list
    generation = generation[:-1]
    generation = sort_by_price(generation)
    supply = 0.0
    indice = 0
    while supply < demand and indice < len(generation):
        supply += generation[indice][0]
        indice += 1
    if supply >= demand:
        # The marginal cost is the price of the last unit added
        marginal_cost_price = generation[indice - 1][1]
        return marginal_cost_price
    else:
        # Not enough supply to meet demand
        return None  # Or raise an exception if preferred

def CrossBorderOptimization(object : dict, constraint:float):

    for i in ['peak', 'offpeak']:
        France_gen = object[i]['FRANCE']
        Spain_gen = object[i]['SPAIN']

        balance_load_fr = analyse_supply(France_gen)
        balance_load_es = analyse_supply(Spain_gen)
        if balance_load_fr > 0 and balance_load_es < 0:
            mylogger.logger.info('Transfer FR to ES')
            fr_power_price = meritorder(France_gen)
            es_power_price = fr_power_price
            if constraint + balance_load_es<0:
                mylogger.logger.error('Cross Border Capa not high enough')
            else:
                mylogger.logger.info(f'{min(np.abs(balance_load_es), constraint)}MW has been transfered.')
        elif balance_load_es > 0 and balance_load_fr < 0:
            mylogger.logger.info('Transfer ES to FR')
            es_power_price = meritorder(Spain_gen)
            fr_power_price = es_power_price
            if constraint + balance_load_fr<0:
                mylogger.logger.error('Cross Border Capa not high enough')
            else:
                mylogger.logger.info(f'{min(np.abs(balance_load_fr), constraint)}MW has been transfered.')

        else:
            mylogger.logger.info('No transfer can be done.')
            fr_power_price = meritorder(France_gen)
            es_power_price = meritorder(Spain_gen)

        mylogger.logger.debug(f'Marginal Cost price in FRANCE {i} : {fr_power_price}')
        mylogger.logger.debug(f'Marginal Cost price in SPAIN {i} : {es_power_price}')



if __name__ == '__main__':
    SP_peak = [(SP_peak_SR_Gene, SP_SR_Mc), (SP_peak_Gas_Gene, SP_Gas_Mc), (SP_peak_WON_Gene, SP_WON_Mc), SP_peak_demand]
    SP_op = [(SP_op_SR_Gene, SP_SR_Mc), (SP_op_Gas_Gene, SP_Gas_Mc), (SP_op_WON_Gene, SP_WON_Mc), SP_op_demand]
    FR_peak = [(FR_peak_NUC_Gene, FR_NUC_Mc), (FR_peak_WON_Gene, FR_WON_Mc), FR_peak_demand]
    FR_op = [(FR_op_NUC_Gene, FR_NUC_Mc), (FR_op_WON_Gene, FR_WON_Mc), FR_op_demand]

    TO_OPTIMIZE = {'peak': {'FRANCE': FR_peak, 'SPAIN': SP_peak}, 'offpeak': {'FRANCE': FR_op, 'SPAIN': SP_op}}
    CrossBorderOptimization(TO_OPTIMIZE, MAX_INJECTION)