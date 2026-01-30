# /// script
# requires-python = ">=3.13"
# dependencies = [
#     "marimo>=0.19.7",
#     "ollama==0.6.1",
#     "pydantic==2.12.5",
#     "pydantic-ai==1.50.0",
#     "python-dotenv==1.2.1",
# ]
# ///

import marimo

__generated_with = "0.19.7"
app = marimo.App(
    width="columns",
    css_file="/usr/local/_marimo/custom.css",
    auto_download=["html"],
)


@app.cell(column=0)
def _(Item, Order):
    order = Order(
        items=[
            Item(
                name="Latte",
                size="Grande",
                quantity=2,
                modifiers=["Oat Milk", "Vanilla Syrup"],
            ),
            Item(
                name="Blueberry Muffin",
                quantity=1,
            ),
        ]
    )
    order.model_dump()
    return


@app.cell
def _():
    from dotenv import load_dotenv

    load_dotenv(".env");
    return


@app.cell
async def _(Order):
    from pydantic_ai import Agent
    from pydantic_ai.models.openai import OpenAIChatModel
    from pydantic_ai.providers.ollama import OllamaProvider
    from pydantic_ai.providers.openai import OpenAIProvider

    ollama_model = OpenAIChatModel(
        model_name='kimi-k2.5:cloud',
        provider=OllamaProvider(base_url='http://localhost:11434/v1'),  
    )

    agent = Agent(
        ollama_model, 
        output_type=Order
    )

    response = await agent.run('Ill take two Lattes and a Bagel.')
    return (response,)


@app.cell
def _(response):
    response.output.model_dump()
    return


@app.cell
def _():
    return


@app.cell(column=1)
def _():
    import marimo as mo
    return


@app.cell
def _():
    from decimal import Decimal
    from typing import Optional
    from pydantic import BaseModel, Field, model_validator, conint

    BASE_PRICES = {
        "Espresso": 3.00,
        "Americano": 3.50,
        "Drip Coffee": 2.50,
        "Latte": 4.50,
        "Cappuccino": 4.50,
        "Flat White": 4.75,
        "Mocha": 5.00,
        "Caramel Macchiato": 5.25,
        "Cold Brew": 4.25,
        "Iced Coffee": 3.00,
        "Frappe (Coffee)": 5.50,
        "Frappe (Mocha)": 5.75,
        "Strawberry Smoothie": 6.00,
        "Chai Latte": 4.75,
        "Matcha Latte": 5.25,
        "Earl Grey Tea": 3.00,
        "Green Tea": 3.00,
        "Hot Chocolate": 4.00,
        "Butter Croissant": 3.50,
        "Blueberry Muffin": 3.75,
        "Bagel": 2.50,
        "Avocado Toast": 7.00,
        "Bacon Gouda Sandwich": 5.50,
    }

    SIZE_ADJUST = {
        "Short": -0.50,
        "Tall": 0.00,
        "Grande": 0.50,
        "Venti": 1.00,
        "Trenta": 1.50,
    }

    MODIFIER_PRICES = {
        "Oat Milk": 0.80,
        "Almond Milk": 0.60,
        "Soy Milk": 0.60,
        "Coconut Milk": 0.70,
        "Breve (Half & Half)": 0.80,
        "Skim Milk": 0.00,
        "Vanilla Syrup": 0.50,
        "Caramel Syrup": 0.50,
        "Hazelnut Syrup": 0.50,
        "Peppermint Syrup": 0.50,
        "Sugar Free Vanilla": 0.50,
        "Classic Syrup": 0.00,
        "Extra Shot": 1.00,
        "Whip Cream": 0.50,
        "No Whip": 0.00,
        "Cold Foam": 1.25,
        "Caramel Drizzle": 0.50,
        "Extra Hot": 0.00,
        "Light Ice": 0.00,
        "No Ice": 0.00,
    }


    class Item(BaseModel):
        name: str
        size: Optional[str] = None
        quantity: conint(ge=1)
        modifiers: list[str] = Field(default_factory=list)

        @property
        def unit_price(self) -> Decimal:
            price = Decimal(str(BASE_PRICES[self.name]))

            if self.size:
                price += Decimal(str(SIZE_ADJUST[self.size]))

            for m in self.modifiers:
                price += Decimal(str(MODIFIER_PRICES[m]))

            return price

        @property
        def line_total(self) -> Decimal:
            return self.unit_price * self.quantity


    class Order(BaseModel):
        items: list[Item]
        total_price: Optional[Decimal] = None

        @model_validator(mode="after")
        def compute_or_validate_total(self):
            computed = sum(i.line_total for i in self.items)

            if self.total_price is None:
                self.total_price = computed
            elif self.total_price != computed:
                raise ValueError(
                    f"Total price mismatch: expected {computed}, got {self.total_price}"
                )

            return self
    return Item, Order


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
