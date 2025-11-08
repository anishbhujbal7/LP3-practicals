#include <iostream>
#include <vector>
using namespace std;

int stepCount = 0; // Global variable to count steps

int fibonacciMemo(int n, vector<int> &dp) {
    stepCount++; // Count each function call as a step

    // Base cases
    if (n <= 1)
        return n;

    // If already computed, use memoized value
    if (dp[n] != -1)
        return dp[n];

    // Recursive calls with memoization
    dp[n] = fibonacciMemo(n - 1, dp) + fibonacciMemo(n - 2, dp);
    return dp[n];
}

int main() {
    int n;
    cout << "Enter n: ";
    cin >> n;

    // Initialize dp array with -1 (means not calculated yet)
    vector<int> dp(n + 1, -1);

    stepCount = 0;
    int fib = fibonacciMemo(n, dp);

    cout << "Fibonacci(" << n << ") = " << fib << endl;
    cout << "Step Count (Memoized Recursive) = " << stepCount << endl;

    return 0;
}
