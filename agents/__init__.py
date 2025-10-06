from agents.c51 import C51Agent
from agents.codac import CODACAgent
from agents.fbrac import FBRACAgent
from agents.iqn import IQNAgent
from agents.value_flows import ValueFlowsAgent

agents = dict(
    c51=C51Agent,
    codac=CODACAgent,
    fbrac=FBRACAgent,
    iqn=IQNAgent,
    value_flows=ValueFlowsAgent,
)
