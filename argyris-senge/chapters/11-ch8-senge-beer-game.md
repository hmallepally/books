# Chapter 8: The Beer Game: Structure Influencing Behavior

> *"When placed in the same system, people, despite their individual differences, tend to produce similar results."*

---

## The Panic of the Retailer

Imagine you are a retailer running a small corner store. One of your products is Lover's Beer, a popular local brew. Typically, you sell about 4 cases of Lover's Beer per week. You keep 12 cases in stock, and you order 4 cases from your wholesaler every Monday to replace what you sold.

Then, in week two, something changes. A popular music video features Lover's Beer. Suddenly, you sell 8 cases. You are thrilled! Your stock drops to 8 cases. To be safe, you order 8 cases from your wholesaler to rebuild your inventory.

In week three, you sell another 8 cases. Your inventory is now down to 4 cases. You realize you need to catch up. But your wholesaler only delivers the 8 cases you ordered last week—there is a two-week shipping delay. Because you want to be safe, you order 12 cases this week.

By week four, you run out of stock. You have 0 cases on the shelf and a backlog of 4 orders from customers who wanted the beer but couldn't get it. You are panicking. You order 20 cases, hoping the wholesaler will deliver them quickly.

What you do not know is that the wholesaler, the distributor, and the brewery are experiencing the exact same panic. Each is doubling and tripling their orders, desperate to catch up with a spike in demand.

This is the setting of the **Beer Game**, a simulation board game developed at the MIT Sloan School of Management in the 1960s to teach system dynamics (Senge, 1990).

## The Mechanics of the Game

The Beer Game consists of four players sitting in a line, representing a simplified supply chain:

```
  [ RETAILER ] ◄─── (2 Weeks) ─── [ WHOLESALER ] ◄─── (2 Weeks) ─── [ DISTRIBUTOR ] ◄─── (2 Weeks) ─── [ BREWERY ]
```

* **The Retailer**: Sells directly to the customer and orders from the Wholesaler.
* **The Wholesaler**: Receives orders from the Retailer and orders from the Distributor.
* **The Distributor**: Receives orders from the Wholesaler and orders from the Brewery.
* **The Brewery**: Receives orders from the Distributor and brews the beer.

There are two critical rules:

1. **Communication Delay**: Orders take 2 weeks to travel up the chain, and shipments take 2 weeks to travel down the chain.
2. **No Direct Communication**: Players are not allowed to talk to each other. The only communication is the order slips they pass to the next player.

## The Bullwhip Effect

In a typical game, the customer demand at the Retailer remains completely stable at 4 cases, spikes to 8 cases in week two, and then *stays at 8 cases* for the rest of the game. It is a single, small step-increase in demand.

Yet, this small change triggers a massive wave of instability up the chain, a phenomenon known in supply chain management as the **Bullwhip Effect**.

```
    Customer Demand:     [ 4 ] ────► [ 8 ] ─────────────────────────► (Stable)
    Retailer Orders:     [ 4 ] ────► [ 8 ] ───► [ 16 ] ───► [ 24 ] ───► [ 0 ]
    Wholesaler Orders:   [ 4 ] ────► [ 8 ] ───► [ 24 ] ───► [ 40 ] ───► [ 0 ]
    Distributor Orders:  [ 4 ] ────► [ 12 ] ───► [ 40 ] ───► [ 80 ] ───► [ 0 ]
    Brewery Production:  [ 4 ] ────► [ 16 ] ───► [ 60 ] ───► [ 120 ] ──► [ 0 ]
```

As orders travel up the chain, the delay causes players to panic. Because they do not receive their shipments immediately, they assume the supplier has failed, and they order more. This over-ordering escalates. By week 15, the Brewery is brewing 120 cases of beer for a system where the actual customer is only buying 8.

Eventually, the massive shipments arrive. The Retailer's backlog disappears, and they are suddenly buried in inventory. They stop ordering. The zero-order propagates up the chain, leaving the entire system paralyzed with excess inventory that takes months to clear.

## Key Systemic Lessons

The Beer Game is not a test of individual competence. Even when played by senior supply chain executives, the results are almost always the same. The game teaches three critical lessons about systems:

### 1. Structure Governs Behavior
When placed in the same system, people tend to produce the same results. The panic, the backlogs, and the excess inventory are not caused by bad players; they are caused by the structure of the system—specifically, the communication delays and the lack of visibility.

### 2. Local Optimization Destroys the System
Each player acted rationally within their own local boundary. They wanted to satisfy their customers and avoid backlog fees. However, because they could not see the whole system, their local optimizations combined to destroy the global system's health.

### 3. The Solution is Structural, Not Behavioral
You cannot solve the Beer Game by telling players to "be more calm" or "work harder." You must change the structure. In modern supply chains, this is done by sharing real-time point-of-sale (POS) data from the Retailer directly with the Wholesaler, Distributor, and Brewery, eliminating the information delays.

---

> ⭐ **STAR Moment: Blaming the Supplier**
>
> *During the Beer Game, players become extremely angry at each other. The Retailer blames the Wholesaler for not delivering beer; the Wholesaler blames the Distributor; the Distributor blames the Brewery. In the post-game review, players are shocked to learn that the Brewery was brewing as fast as possible, and the customer demand was stable. They learn Senge's core lesson: "There is no 'other.' You and your supplier are part of a single system. When you blame them, you are blaming a part of yourself."*
