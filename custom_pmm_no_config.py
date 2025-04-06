import logging
from decimal import Decimal
from typing import List
import numpy as np
from collections import deque
from hummingbot.core.data_type.common import OrderType, PriceType, TradeType
from hummingbot.core.data_type.order_candidate import OrderCandidate
from hummingbot.core.event.events import OrderFilledEvent
from hummingbot.strategy.script_strategy_base import ScriptStrategyBase


class CustomPMM(ScriptStrategyBase):
    bid_spread = 0.01
    ask_spread = 0.01
    order_refresh_time = 30
    order_amount = 0.01
    create_timestamp = 0
    trading_pair = "BTC-USDT"
    exchange = "binance_paper_trade"
    price_source = PriceType.MidPrice
    ref_prices: deque = deque(maxlen=10)

    markets = {exchange: {trading_pair}}

    def on_tick(self):
        ref_price = self.connectors[self.exchange].get_price_by_type(
            self.trading_pair, self.price_source)
        self.ref_prices.append(Decimal(ref_price))

        if len(self.ref_prices) < self.ref_prices.maxlen:
            self.logger().info(f"Collecting the mid prices: {len(self.ref_prices)}/{self.ref_prices.maxlen}")
            return

        # --- Volatility-based Scaling ---
        volatility = self.calculate_volatility()
        target_vol = 0.005
        sensitivity = 100 
        vol_diff = volatility - target_vol
        sigmoid = 1 / (1 + np.exp(-vol_diff * sensitivity))
        scale = 1.5 - sigmoid
        base_amount = Decimal(self.order_amount * scale)
        dynamic_bid_spread = Decimal(self.bid_spread * (2 - scale))
        dynamic_ask_spread = Decimal(self.ask_spread * (2 - scale))

        # --- Inventory Skew Logic ---
        mid_price = Decimal(ref_price)
        base, quote = self.trading_pair.split("-")
        btc_balance = self.connectors[self.exchange].get_balance(base)
        usdt_balance = self.connectors[self.exchange].get_balance(quote)


        btc_value = btc_balance * mid_price
        total_value = btc_value + usdt_balance

        if total_value == 0:
            self.logger().warning("Total portfolio value is 0. Skipping tick.")
            return

        target_ratio = Decimal("0.5")
        current_ratio = btc_value / total_value
        inventory_skew = current_ratio - target_ratio

        skew_strength = Decimal("1.5")
        buy_amount = base_amount * (Decimal("1.0") - skew_strength * inventory_skew)
        sell_amount = base_amount * (Decimal("1.0") + skew_strength * inventory_skew)

        min_order = Decimal("0.0001")
        buy_amount = max(buy_amount, min_order)
        sell_amount = max(sell_amount, min_order)

        # --- Order Placement ---
        if self.create_timestamp <= self.current_timestamp:
            self.cancel_all_orders()
            proposal = self.create_proposal(mid_price, dynamic_bid_spread, dynamic_ask_spread, buy_amount, sell_amount)
            adjusted_proposal = self.adjust_proposal_to_budget(proposal)
            self.place_orders(adjusted_proposal)
            self.create_timestamp = self.current_timestamp + self.order_refresh_time

    def calculate_volatility(self):
        prices = np.array([float(p) for p in self.ref_prices])
        if np.any(prices <= 0):
            return 0.0001   
        returns = np.diff(np.log(prices))
        volatility = np.std(returns)
        if np.isnan(volatility) or volatility == 0:
            volatility = 0.0001
        return volatility

    def create_proposal(self, ref_price: Decimal, bid_spread: Decimal, ask_spread: Decimal,
                        buy_amount: Decimal, sell_amount: Decimal) -> List[OrderCandidate]:
        buy_price = ref_price * Decimal(1 - (bid_spread) / 100)
        sell_price = ref_price * Decimal(1 + (ask_spread) / 100)

        buy_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                   order_side=TradeType.BUY, amount=buy_amount, price=buy_price)

        sell_order = OrderCandidate(trading_pair=self.trading_pair, is_maker=True, order_type=OrderType.LIMIT,
                                    order_side=TradeType.SELL, amount=sell_amount, price=sell_price)

        return [buy_order, sell_order]

    def adjust_proposal_to_budget(self, proposal: List[OrderCandidate]) -> List[OrderCandidate]:
        proposal_adjusted = self.connectors[self.exchange].budget_checker.adjust_candidates(proposal, all_or_none=True)
        return proposal_adjusted

    def place_orders(self, proposal: List[OrderCandidate]) -> None:
        for order in proposal:
            self.place_order(connector_name=self.exchange, order=order)

    def place_order(self, connector_name: str, order: OrderCandidate):
        if order.order_side == TradeType.SELL:
            self.sell(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                      order_type=order.order_type, price=order.price)
        elif order.order_side == TradeType.BUY:
            self.buy(connector_name=connector_name, trading_pair=order.trading_pair, amount=order.amount,
                     order_type=order.order_type, price=order.price)

    def cancel_all_orders(self):
        for order in self.get_active_orders(connector_name=self.exchange):
            self.cancel(self.exchange, order.trading_pair, order.client_order_id)

    def did_fill_order(self, event: OrderFilledEvent):
        msg = (f"{event.trade_type.name} {round(event.amount, 2)} {event.trading_pair} {self.exchange} at {round(event.price, 2)}")
        self.log_with_clock(logging.INFO, msg)
        self.notify_hb_app_with_timestamp(msg)
