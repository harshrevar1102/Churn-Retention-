-- Monthly revenue, active customers and orders (for the dashboard trend chart)
SELECT
    strftime('%Y-%m', InvoiceDate)  AS month,
    ROUND(SUM(Revenue), 2)          AS revenue,
    COUNT(DISTINCT CustomerID)      AS active_customers,
    COUNT(DISTINCT InvoiceNo)       AS orders
FROM sales
GROUP BY month
ORDER BY month;
