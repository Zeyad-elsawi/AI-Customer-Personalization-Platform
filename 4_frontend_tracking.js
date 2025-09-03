/**
 * Frontend JavaScript Tracking for AI Customer Personalization Platform
 * Tracks user behavior and sends events to the backend API
 */

class CustomerPersonalizationTracker {
    constructor(config = {}) {
        this.apiUrl = config.apiUrl || 'http://localhost:8000';
        this.userId = config.userId || this.generateUserId();
        this.sessionId = config.sessionId || this.generateSessionId();
        this.debug = config.debug || false;
        this.batchSize = config.batchSize || 10;
        this.flushInterval = config.flushInterval || 5000; // 5 seconds
        
        this.eventQueue = [];
        this.isOnline = navigator.onLine;
        this.lastActivity = Date.now();
        
        this.init();
    }
    
    init() {
        this.log('🚀 Customer Personalization Tracker initialized');
        
        // Set up event listeners
        this.setupEventListeners();
        
        // Start periodic flush
        this.startPeriodicFlush();
        
        // Track page load
        this.trackPageView();
        
        // Track user activity
        this.trackUserActivity();
        
        // Handle page unload
        this.setupPageUnload();
    }
    
    generateUserId() {
        // Try to get existing user ID from localStorage
        let userId = localStorage.getItem('cp_user_id');
        if (!userId) {
            userId = 'user_' + Math.random().toString(36).substr(2, 9);
            localStorage.setItem('cp_user_id', userId);
        }
        return userId;
    }
    
    generateSessionId() {
        return 'sess_' + Math.random().toString(36).substr(2, 9) + '_' + Date.now();
    }
    
    log(message, data = null) {
        if (this.debug) {
            console.log(`[CP Tracker] ${message}`, data);
        }
    }
    
    setupEventListeners() {
        // Track clicks on product links
        document.addEventListener('click', (e) => {
            const link = e.target.closest('a[href*="/product/"], a[href*="/item/"], a[href*="/p/"]');
            if (link) {
                const productId = this.extractProductId(link.href);
                if (productId) {
                    this.trackProductView(productId, link.href);
                }
            }
        });
        
        // Track form submissions (checkout, contact, etc.)
        document.addEventListener('submit', (e) => {
            const form = e.target;
            if (form.classList.contains('checkout-form') || form.id === 'checkout') {
                this.trackCheckoutStarted();
            }
        });
        
        // Track scroll depth
        this.trackScrollDepth();
        
        // Track time on page
        this.trackTimeOnPage();
    }
    
    extractProductId(url) {
        // Extract product ID from various URL patterns
        const patterns = [
            /\/product\/([^\/\?]+)/,
            /\/item\/([^\/\?]+)/,
            /\/p\/([^\/\?]+)/,
            /product_id=([^&]+)/,
            /id=([^&]+)/
        ];
        
        for (const pattern of patterns) {
            const match = url.match(pattern);
            if (match) {
                return match[1];
            }
        }
        
        return null;
    }
    
    trackUserActivity() {
        // Track mouse movement and clicks to detect engagement
        let activityTimeout;
        
        const resetActivityTimeout = () => {
            clearTimeout(activityTimeout);
            activityTimeout = setTimeout(() => {
                this.lastActivity = Date.now();
            }, 1000);
        };
        
        document.addEventListener('mousemove', resetActivityTimeout);
        document.addEventListener('click', resetActivityTimeout);
        document.addEventListener('keypress', resetActivityTimeout);
        document.addEventListener('scroll', resetActivityTimeout);
    }
    
    trackScrollDepth() {
        let maxScrollDepth = 0;
        let scrollDepthTracked = false;
        
        const trackScroll = () => {
            const scrollTop = window.pageYOffset || document.documentElement.scrollTop;
            const docHeight = document.documentElement.scrollHeight - window.innerHeight;
            const scrollPercent = Math.round((scrollTop / docHeight) * 100);
            
            if (scrollPercent > maxScrollDepth) {
                maxScrollDepth = scrollPercent;
                
                // Track milestone scroll depths
                if (scrollPercent >= 25 && !scrollDepthTracked) {
                    this.trackEvent('scroll_depth', { depth: 25 });
                    scrollDepthTracked = true;
                } else if (scrollPercent >= 50) {
                    this.trackEvent('scroll_depth', { depth: 50 });
                } else if (scrollPercent >= 75) {
                    this.trackEvent('scroll_depth', { depth: 75 });
                } else if (scrollPercent >= 90) {
                    this.trackEvent('scroll_depth', { depth: 90 });
                }
            }
        };
        
        window.addEventListener('scroll', trackScroll);
    }
    
    trackTimeOnPage() {
        const startTime = Date.now();
        
        // Track time milestones
        setTimeout(() => this.trackEvent('time_on_page', { seconds: 30 }), 30000);
        setTimeout(() => this.trackEvent('time_on_page', { seconds: 60 }), 60000);
        setTimeout(() => this.trackEvent('time_on_page', { seconds: 120 }), 120000);
        
        // Track total time on page unload
        window.addEventListener('beforeunload', () => {
            const totalTime = Math.round((Date.now() - startTime) / 1000);
            this.trackEvent('session_end', { total_time_seconds: totalTime });
        });
    }
    
    setupPageUnload() {
        // Send remaining events before page unload
        window.addEventListener('beforeunload', () => {
            this.flushEvents(true); // Force synchronous send
        });
        
        // Handle visibility change (tab switching)
        document.addEventListener('visibilitychange', () => {
            if (document.hidden) {
                this.trackEvent('page_hidden');
            } else {
                this.trackEvent('page_visible');
            }
        });
    }
    
    startPeriodicFlush() {
        setInterval(() => {
            if (this.eventQueue.length > 0) {
                this.flushEvents();
            }
        }, this.flushInterval);
    }
    
    // Core tracking methods
    trackEvent(eventType, properties = {}) {
        const event = {
            event_type: eventType,
            user_id: this.userId,
            session_id: this.sessionId,
            timestamp: new Date().toISOString(),
            page_url: window.location.href,
            referrer: document.referrer,
            device: this.getDeviceType(),
            country: this.getCountry(),
            user_segment: this.getUserSegment(),
            ...properties
        };
        
        this.eventQueue.push(event);
        this.log(`Event tracked: ${eventType}`, event);
        
        // Flush if queue is full
        if (this.eventQueue.length >= this.batchSize) {
            this.flushEvents();
        }
    }
    
    trackPageView() {
        this.trackEvent('page_view', {
            page_title: document.title,
            page_path: window.location.pathname,
            page_search: window.location.search
        });
    }
    
    trackProductView(productId, productUrl = null, productData = {}) {
        this.trackEvent('product_view', {
            product_id: productId,
            product_url: productUrl || window.location.href,
            product_name: productData.name || this.extractProductName(),
            product_price: productData.price || this.extractProductPrice(),
            product_category: productData.category || this.extractProductCategory(),
            product_brand: productData.brand || this.extractProductBrand()
        });
    }
    
    trackAddToCart(productId, quantity = 1, price = null, productData = {}) {
        this.trackEvent('add_to_cart', {
            product_id: productId,
            quantity: quantity,
            price: price || this.extractProductPrice(),
            product_name: productData.name || this.extractProductName(),
            product_category: productData.category || this.extractProductCategory(),
            product_brand: productData.brand || this.extractProductBrand()
        });
    }
    
    trackRemoveFromCart(productId, quantity = 1, price = null) {
        this.trackEvent('remove_from_cart', {
            product_id: productId,
            quantity: quantity,
            price: price || this.extractProductPrice()
        });
    }
    
    trackCheckoutStarted() {
        this.trackEvent('checkout_started', {
            checkout_url: window.location.href
        });
    }
    
    trackPurchase(orderId, totalAmount, items = []) {
        this.trackEvent('purchase', {
            order_id: orderId,
            total_amount: totalAmount,
            items: items,
            currency: 'USD'
        });
    }
    
    // Helper methods for extracting product information
    extractProductName() {
        // Try to extract product name from page
        const selectors = [
            'h1.product-title',
            'h1.product-name',
            '.product-title',
            '.product-name',
            'h1',
            '.title'
        ];
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                return element.textContent.trim();
            }
        }
        
        return null;
    }
    
    extractProductPrice() {
        // Try to extract product price from page
        const selectors = [
            '.price',
            '.product-price',
            '.current-price',
            '[data-price]',
            '.price-current'
        ];
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                const priceText = element.textContent || element.getAttribute('data-price');
                const price = parseFloat(priceText.replace(/[^0-9.]/g, ''));
                if (!isNaN(price)) {
                    return price;
                }
            }
        }
        
        return null;
    }
    
    extractProductCategory() {
        // Try to extract product category from page
        const selectors = [
            '.breadcrumb a:last-child',
            '.category',
            '.product-category',
            '[data-category]'
        ];
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                return element.textContent.trim();
            }
        }
        
        return null;
    }
    
    extractProductBrand() {
        // Try to extract product brand from page
        const selectors = [
            '.brand',
            '.product-brand',
            '[data-brand]'
        ];
        
        for (const selector of selectors) {
            const element = document.querySelector(selector);
            if (element) {
                return element.textContent.trim();
            }
        }
        
        return null;
    }
    
    getDeviceType() {
        const width = window.innerWidth;
        if (width < 768) return 'mobile';
        if (width < 1024) return 'tablet';
        return 'desktop';
    }
    
    getCountry() {
        // Try to get country from various sources
        return localStorage.getItem('cp_country') || 'unknown';
    }
    
    getUserSegment() {
        // Try to get user segment from localStorage or determine from behavior
        return localStorage.getItem('cp_user_segment') || 'unknown';
    }
    
    // API communication
    async flushEvents(synchronous = false) {
        if (this.eventQueue.length === 0) return;
        
        const events = [...this.eventQueue];
        this.eventQueue = [];
        
        this.log(`Flushing ${events.length} events`);
        
        if (synchronous) {
            // Use sendBeacon for synchronous requests (page unload)
            for (const event of events) {
                const blob = new Blob([JSON.stringify(event)], { type: 'application/json' });
                navigator.sendBeacon(`${this.apiUrl}/events`, blob);
            }
        } else {
            // Use fetch for asynchronous requests
            try {
                for (const event of events) {
                    await fetch(`${this.apiUrl}/events`, {
                        method: 'POST',
                        headers: {
                            'Content-Type': 'application/json',
                        },
                        body: JSON.stringify(event)
                    });
                }
                this.log(`Successfully sent ${events.length} events`);
            } catch (error) {
                this.log(`Error sending events: ${error.message}`);
                // Re-queue events on error
                this.eventQueue.unshift(...events);
            }
        }
    }
    
    // Public API methods
    setUserId(userId) {
        this.userId = userId;
        localStorage.setItem('cp_user_id', userId);
    }
    
    setUserSegment(segment) {
        localStorage.setItem('cp_user_segment', segment);
    }
    
    setCountry(country) {
        localStorage.setItem('cp_country', country);
    }
    
    // E-commerce integration helpers
    integrateWithShopify() {
        // Shopify-specific integration
        if (typeof Shopify !== 'undefined') {
            // Track Shopify cart events
            document.addEventListener('DOMContentLoaded', () => {
                // Track cart updates
                const cartForm = document.querySelector('form[action*="/cart"]');
                if (cartForm) {
                    cartForm.addEventListener('submit', (e) => {
                        const formData = new FormData(cartForm);
                        const items = formData.getAll('updates[]');
                        // Track cart updates
                    });
                }
            });
        }
    }
    
    integrateWithWooCommerce() {
        // WooCommerce-specific integration
        if (typeof wc_add_to_cart_params !== 'undefined') {
            // Track WooCommerce add to cart events
            document.addEventListener('click', (e) => {
                if (e.target.classList.contains('single_add_to_cart_button')) {
                    const productId = e.target.getAttribute('data-product_id');
                    const quantity = e.target.form?.querySelector('input[name="quantity"]')?.value || 1;
                    this.trackAddToCart(productId, quantity);
                }
            });
        }
    }
}

// Auto-initialize if not already done
if (typeof window.cpTracker === 'undefined') {
    window.cpTracker = new CustomerPersonalizationTracker({
        debug: true, // Set to false in production
        apiUrl: 'http://localhost:8000'
    });
}

// Export for module systems
if (typeof module !== 'undefined' && module.exports) {
    module.exports = CustomerPersonalizationTracker;
}
