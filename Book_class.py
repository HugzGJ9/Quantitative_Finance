from Asset_class import asset_BS
from Options_class import  Option_eu, Option_prem_gen

class Book():
    def __init__(self, options_basket:list, asset=None)->None:
        self.basket = options_basket
        self.asset = asset
        return
    def append(self, option:(Option_eu, Option_prem_gen))->None:
        self.basket.append(option)
        return
    def delta_hedge(self, asset:asset_BS):
        if not isinstance(asset, asset_BS):
            raise TypeError("asset must be of type asset_BS")
        self.asset = asset
        return
    #def remove(self, ):
    def option_price_close_formulae(self):
        return sum([option.option_price_close_formulae() if isinstance(option, (Option_eu, Option_prem_gen)) else 0 for option in self.basket])
    def Delta_DF(self):
        hedge = self.asset.quantity if self.asset != None else 0
        return sum([option.Delta_DF() for option in self.basket]) + hedge
    def Gamma_DF(self):
        return sum([option.Gamma_DF() for option in self.basket])
    def Vega_DF(self):
        return sum([option.Vega_DF() for option in self.basket])
    def Theta_DF(self):
        return sum([option.Theta_DF() for option in self.basket])
    def simu_asset(self, time):
        list_asset = list(set([x.asset for x in self.basket]))
        for item in list_asset:
            item.simu_asset(time)