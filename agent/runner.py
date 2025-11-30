from agent.core import Agent
from agent.tools import predict_yield_tool, predict_aqi_tool


def build_agent() -> Agent:
    agent = Agent(
        name="AgriEnvDecisionAgent",
        description="Predicts crop yield and AQI and gives basic advisory messages."
    )
    agent.add_tool("predict_yield", predict_yield_tool)
    agent.add_tool("predict_aqi", predict_aqi_tool)
    return agent


def main():
    agent = build_agent()
    print(f"Agent: {agent.name}")
    print("Tools:", agent.list_tools())
    print("Type 'yield' or 'aqi' or 'quit'.")

    while True:
        choice = input("Command> ").strip().lower()
        if choice in ("quit", "exit"):
            break

        if choice == "yield":
            soil_pH = float(input("soil_pH: "))
            N = float(input("N: "))
            P = float(input("P: "))
            K = float(input("K: "))
            rainfall = float(input("rainfall: "))
            temperature = float(input("temperature: "))
            y = agent.call_tool(
                "predict_yield",
                soil_pH=soil_pH,
                N=N,
                P=P,
                K=K,
                rainfall=rainfall,
                temperature=temperature
            )
            print(f"Predicted Yield: {y:.3f}")
            if y < 3:
                print("Advisory: ⚠ Low yield – consider adjusting fertilizer and irrigation.")
            else:
                print("Advisory: ✅ Yield is satisfactory.")
        elif choice == "aqi":
            pm25 = float(input("PM2.5: "))
            pm10 = float(input("PM10: "))
            co = float(input("CO: "))
            no2 = float(input("NO2: "))
            temp = float(input("Temp: "))
            humidity = float(input("Humidity: "))
            aqi = agent.call_tool(
                "predict_aqi",
                pm25=pm25,
                pm10=pm10,
                co=co,
                no2=no2,
                temp=temp,
                humidity=humidity
            )
            print(f"Predicted AQI: {aqi:.1f}")
            if aqi > 150:
                print("Advisory: ⚠ Air quality unhealthy – limit outdoor exposure.")
            else:
                print("Advisory: ✅ Air quality acceptable.")
        else:
            print("Unknown command. Use 'yield', 'aqi', or 'quit'.")


if __name__ == "__main__":
    main()

