def process(cfgs):
    # 1st make clear the tool we want to use
    tool = cfgs.get("tool", None)
    assert tool is not None, "Tool must be specified in the configuration."