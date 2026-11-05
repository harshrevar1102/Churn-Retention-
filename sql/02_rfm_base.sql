-- Raw Recency / Frequency / Monetary per customer as of :snapshot
SELECT
    CustomerID,
    CAST(julianday(:snapshot) - julianday(MAX(InvoiceDate)) AS INTEGER) AS recency_days,
    COUNT(DISTINCT InvoiceNo)  AS frequency,
    ROUND(SUM(Revenue), 2)     AS monetary
FROM sales
WHERE InvoiceDate <= :snapshot
GROUP BY CustomerID;
