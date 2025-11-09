#include <iostream>
#include <vector>
#include <algorithm>
#include <iomanip> // for setprecision
using namespace std;

struct Item {
    int id;
    double weight;
    double profit;
    double ratio;
};

// Comparison: sort by profit/weight ratio descending
bool compare(Item a, Item b) {
    return a.ratio > b.ratio;
}

void fractionalKnapsack(vector<Item> &items, double capacity) {
    // Step 1: Compute profit/weight ratio for each item
    for (auto &item : items)
        item.ratio = item.profit / item.weight;

    // Step 2: Sort items by ratio (descending)
    sort(items.begin(), items.end(), compare);

    double totalProfit = 0.0;
    double remainingCapacity = capacity;

    cout << fixed << setprecision(2);
    cout << "\nItem Selection Steps:\n";
    cout << "-----------------------------------------------------------\n";
    cout << "Item\tWeight\tProfit\tTaken(%)\tProfit Added\n";
    cout << "-----------------------------------------------------------\n";

    // Step 3: Take items greedily
    for (auto &item : items) {
        if (remainingCapacity == 0)
            break;

        double takenFraction = 0.0;
        double profitAdded = 0.0;

        if (item.weight <= remainingCapacity) {
            // Take the whole item
            remainingCapacity -= item.weight;
            profitAdded = item.profit;
            takenFraction = 1.0;
        } else {
            // Take fractional part
            takenFraction = remainingCapacity / item.weight;
            profitAdded = item.profit * takenFraction;
            remainingCapacity = 0;
        }

        totalProfit += profitAdded;

        cout << item.id << "\t"
             << item.weight << "\t"
             << item.profit << "\t"
             << takenFraction * 100 << "%\t\t"
             << profitAdded << "\n";
    }

    cout << "-----------------------------------------------------------\n";
    cout << "Total Profit = " << totalProfit << endl;
}

int main() {
    int n;
    double capacity;

    cout << "Enter number of items: ";
    cin >> n;

    vector<Item> items(n);

    cout << "Enter profit and weight of each item:\n";
    for (int i = 0; i < n; i++) {
        items[i].id = i + 1;
        cin >> items[i].profit >> items[i].weight;
    }

    cout << "Enter Knapsack Capacity: ";
    cin >> capacity;

    fractionalKnapsack(items, capacity);

    return 0;
}



// Enter number of items: 3
// Enter profit and weight of each item:
// 60 10
// 100 20
// 120 30
// Enter Knapsack Capacity: 50


// | Step             | Operation      | Time Complexity |
// | ---------------- | -------------- | --------------- |
// | Calculate ratios | O(n)           |                 |
// | Sort items       | O(n log n)     |                 |
// | Pick items       | O(n)           |                 |
// | **Total**        | **O(n log n)** |                 |
