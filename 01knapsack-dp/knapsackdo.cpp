#include <iostream>
#include <vector>
#include <iomanip>
using namespace std;

// Function to solve 0/1 Knapsack using Dynamic Programming
int knapsackDP(int W, vector<int> &weights, vector<int> &profits, int n) {
    vector<vector<int>> dp(n + 1, vector<int>(W + 1, 0));

    // Build table dp[][] bottom-up
    for (int i = 1; i <= n; i++) {
        for (int w = 1; w <= W; w++) {
            if (weights[i - 1] <= w) {
                dp[i][w] = max(
                    profits[i - 1] + dp[i - 1][w - weights[i - 1]],
                    dp[i - 1][w]
                );
            } else {
                dp[i][w] = dp[i - 1][w];
            }
        }
    }

    // Print DP table (optional, for understanding)
    cout << "\nDP Table:\n";
    for (int i = 0; i <= n; i++) {
        for (int w = 0; w <= W; w++) {
            cout << setw(4) << dp[i][w];
        }
        cout << endl;
    }

    return dp[n][W];
}

int main() {
    int n, W;
    cout << "Enter number of items: ";
    cin >> n;

    vector<int> profits(n), weights(n);

    cout << "Enter profit and weight of each item:\n";
    for (int i = 0; i < n; i++) {
        cin >> profits[i] >> weights[i];
    }

    cout << "Enter knapsack capacity: ";
    cin >> W;

    int maxProfit = knapsackDP(W, weights, profits, n);

    cout << "\nMaximum Profit = " << maxProfit << endl;

    return 0;
}




// Enter number of items: 4
// Enter profit and weight of each item:
// 1 1
// 2 3
// 5 4
// 6 5
// Enter knapsack capacity: 8


// | Type  | Complexity                                          |
// | ----- | --------------------------------------------------- |
// | Time  | **O(n × W)**                                        |
// | Space | **O(n × W)** (can be optimized to O(W) using 1D DP) |
