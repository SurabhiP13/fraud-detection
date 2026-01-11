// API configuration
const API_BASE_URL = 'http://localhost:8000';

// State
let fraudOnlyFilter = false;

// DOM elements
const startBtn = document.getElementById('startBtn');
const refreshBtn = document.getElementById('refreshBtn');
const addTransactionBtn = document.getElementById('addTransactionBtn');
const transactionsContainer = document.getElementById('transactionsContainer');
const fraudOnlyCheckbox = document.getElementById('fraudOnlyFilter');
const modal = document.getElementById('addTransactionModal');
const closeModal = document.querySelector('.close');
const transactionForm = document.getElementById('transactionForm');

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    loadStats();
    loadTransactions();
    setupEventListeners();
});

function setupEventListeners() {
    startBtn.addEventListener('click', startProcessing);
    refreshBtn.addEventListener('click', refreshData);
    addTransactionBtn.addEventListener('click', openModal);
    closeModal.addEventListener('click', closeModalHandler);
    fraudOnlyCheckbox.addEventListener('change', handleFilterChange);
    transactionForm.addEventListener('submit', handleSubmitTransaction);
    
    window.addEventListener('click', (e) => {
        if (e.target === modal) {
            closeModalHandler();
        }
    });
}

async function startProcessing() {
    try {
        startBtn.disabled = true;
        startBtn.innerHTML = '⏳ Processing...';
        
        const response = await fetch(`${API_BASE_URL}/api/transactions/start`, {
            method: 'POST'
        });
        
        if (!response.ok) throw new Error('Failed to start processing');
        
        const data = await response.json();
        
        showNotification('Transaction processing started!', 'success');
        startBtn.innerHTML = '✅ Processing Started';
        
        setTimeout(() => {
            startBtn.disabled = false;
            startBtn.innerHTML = '🚀 Start Transaction Processing';
            refreshData();
        }, 2000);
        
    } catch (error) {
        console.error('Error starting processing:', error);
        showNotification('Failed to start processing', 'error');
        startBtn.disabled = false;
        startBtn.innerHTML = '🚀 Start Transaction Processing';
    }
}

async function loadStats() {
    try {
        const response = await fetch(`${API_BASE_URL}/api/stats`);
        if (!response.ok) throw new Error('Failed to load stats');
        
        const stats = await response.json();
        
        document.getElementById('totalTransactions').textContent = stats.total_transactions;
        document.getElementById('fraudTransactions').textContent = stats.fraud_transactions;
        document.getElementById('legitimateTransactions').textContent = stats.legitimate_transactions;
        document.getElementById('fraudPercentage').textContent = stats.fraud_percentage.toFixed(1) + '%';
        
    } catch (error) {
        console.error('Error loading stats:', error);
    }
}

async function loadTransactions() {
    try {
        transactionsContainer.innerHTML = '<div class="loading">Loading transactions...</div>';
        
        const url = `${API_BASE_URL}/api/transactions?fraud_only=${fraudOnlyFilter}`;
        const response = await fetch(url);
        
        if (!response.ok) throw new Error('Failed to load transactions');
        
        const transactions = await response.json();
        
        if (transactions.length === 0) {
            transactionsContainer.innerHTML = `
                <div class="no-transactions">
                    No transactions found. Click "Add Test Transaction" to create some!
                </div>
            `;
            return;
        }
        
        renderTransactions(transactions);
        
    } catch (error) {
        console.error('Error loading transactions:', error);
        transactionsContainer.innerHTML = `
            <div class="no-transactions">
                Failed to load transactions. Make sure the API is running.
            </div>
        `;
    }
}

function renderTransactions(transactions) {
    const html = transactions.map(transaction => {
        const fraudClass = transaction.is_fraud ? 'fraud' : '';
        const fraudBadge = transaction.is_fraud ? 
            '<span class="fraud-badge fraud">🚨 FRAUD DETECTED</span>' :
            '<span class="fraud-badge legitimate">✅ LEGITIMATE</span>';
        
        return `
            <div class="transaction-item ${fraudClass}">
                <div class="transaction-header">
                    <span class="transaction-id">Transaction #${transaction.id}</span>
                    ${fraudBadge}
                </div>
                <div class="transaction-details">
                    <div class="detail-item">
                        <span class="detail-label">Amount</span>
                        <span class="detail-value amount">$${transaction.amount.toFixed(2)}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Merchant</span>
                        <span class="detail-value">${transaction.merchant}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Category</span>
                        <span class="detail-value">${transaction.category}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">User ID</span>
                        <span class="detail-value">${transaction.user_id}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Location</span>
                        <span class="detail-value">${transaction.location || 'N/A'}</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Fraud Score</span>
                        <span class="detail-value">${(transaction.fraud_score * 100).toFixed(1)}%</span>
                    </div>
                    <div class="detail-item">
                        <span class="detail-label">Timestamp</span>
                        <span class="detail-value">${new Date(transaction.timestamp).toLocaleString()}</span>
                    </div>
                </div>
            </div>
        `;
    }).join('');
    
    transactionsContainer.innerHTML = html;
}

function handleFilterChange(e) {
    fraudOnlyFilter = e.target.checked;
    loadTransactions();
}

function refreshData() {
    loadStats();
    loadTransactions();
    showNotification('Data refreshed!', 'success');
}

function openModal() {
    modal.style.display = 'block';
}

function closeModalHandler() {
    modal.style.display = 'none';
    transactionForm.reset();
}

async function handleSubmitTransaction(e) {
    e.preventDefault();
    
    const transaction = {
        amount: parseFloat(document.getElementById('amount').value),
        merchant: document.getElementById('merchant').value,
        category: document.getElementById('category').value,
        location: document.getElementById('location').value || null,
        user_id: document.getElementById('userId').value
    };
    
    try {
        const response = await fetch(`${API_BASE_URL}/api/transactions`, {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(transaction)
        });
        
        if (!response.ok) throw new Error('Failed to create transaction');
        
        const result = await response.json();
        
        showNotification(
            result.is_fraud ? 
                '🚨 Transaction created - FRAUD DETECTED!' : 
                '✅ Transaction created - Legitimate',
            result.is_fraud ? 'warning' : 'success'
        );
        
        closeModalHandler();
        refreshData();
        
    } catch (error) {
        console.error('Error creating transaction:', error);
        showNotification('Failed to create transaction', 'error');
    }
}

function showNotification(message, type = 'info') {
    // Simple notification system
    const notification = document.createElement('div');
    notification.style.cssText = `
        position: fixed;
        top: 20px;
        right: 20px;
        padding: 15px 20px;
        background: ${type === 'error' ? '#f5576c' : type === 'warning' ? '#fee140' : '#4facfe'};
        color: ${type === 'warning' ? '#333' : 'white'};
        border-radius: 8px;
        box-shadow: 0 4px 12px rgba(0,0,0,0.2);
        z-index: 10000;
        font-weight: 600;
        animation: slideIn 0.3s ease;
    `;
    notification.textContent = message;
    document.body.appendChild(notification);
    
    setTimeout(() => {
        notification.style.animation = 'slideOut 0.3s ease';
        setTimeout(() => notification.remove(), 300);
    }, 3000);
}

// Auto-refresh every 10 seconds
setInterval(() => {
    loadStats();
    loadTransactions();
}, 10000);
