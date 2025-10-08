from flask import Flask,render_template,request,jsonify,session
import pickle
import numpy as np
import joblib
from datetime import datetime
import pandas as pd
import secrets

# Load all models and data
popular_df = pickle.load(open('../models/popular_new.pkl','rb'))
pt = pickle.load(open('../models/pt_new.pkl','rb'))
books = pickle.load(open('../models/books_new.pkl','rb'))
similarity_scores = pickle.load(open('../models/similarity_scores_new.pkl','rb'))

# Load ML models
svd_model = joblib.load('../models/svd_model.pkl')
ml_data_filtered = pickle.load(open('../models/ml_data_filtered.pkl', 'rb'))
trainset = pickle.load(open('../models/trainset.pkl', 'rb'))

# ======================= GNN MODEL SETUP =======================
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GATConv, GATv2Conv, SAGEConv, GCNConv
from torch_geometric.data import Data
import warnings
warnings.filterwarnings('ignore')

class AdvancedBookRecommenderGNN(nn.Module):
    """Advanced GNN model for book recommendations - Compatible with saved model"""
    def __init__(self, num_users=12366, num_books=1616, embedding_dim=64, dropout=0.3):
        super(AdvancedBookRecommenderGNN, self).__init__()
        
        self.num_users = num_users
        self.num_books = num_books
        self.embedding_dim = embedding_dim
        self.dropout = dropout
        
        # Embeddings to match saved model exactly
        self.user_embedding = nn.Embedding(num_users, embedding_dim)
        self.book_embedding = nn.Embedding(num_books, embedding_dim)
        
        # Graph layers to match saved model exactly
        # GAT: 64 -> 32*4=128 (4 heads, 32 each) - using original GATConv
        self.gat1 = GATConv(embedding_dim, 32, heads=4, dropout=dropout)
        self.bn1 = nn.ModuleDict({'module': nn.BatchNorm1d(128)})
        
        # SAGE: 128 -> 64  
        self.sage = SAGEConv(128, 64)
        self.bn2 = nn.ModuleDict({'module': nn.BatchNorm1d(64)})
        
        # GCN: 64 -> 32
        self.gcn = GCNConv(64, 32)
        self.bn3 = nn.ModuleDict({'module': nn.BatchNorm1d(32)})
        
        # Bias terms
        self.user_bias = nn.Embedding(num_users, 1)
        self.book_bias = nn.Embedding(num_books, 1)
        self.global_bias = nn.Parameter(torch.zeros(1))
        
        # Predictor matching saved model exactly: [192 -> 64 -> 32 -> 32 -> 1]
        # Input: user_emb(64) + book_emb(64) + gcn_output(32) + user_bias(1) + book_bias(1) + global_bias = ~192
        self.predictor = nn.Sequential(
            nn.Linear(192, 64),  # 192 -> 64
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(64, 32),   # 64 -> 32
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(32, 32),   # 32 -> 32
            nn.ReLU(),
            nn.Linear(32, 1)     # 32 -> 1
        )
        
        self.init_weights()
    
    def init_weights(self):
        """Initialize weights"""
        nn.init.xavier_uniform_(self.user_embedding.weight)
        nn.init.xavier_uniform_(self.book_embedding.weight)
        nn.init.zeros_(self.user_bias.weight)
        nn.init.zeros_(self.book_bias.weight)
        
        for layer in self.predictor:
            if isinstance(layer, nn.Linear):
                nn.init.xavier_uniform_(layer.weight)
                nn.init.zeros_(layer.bias)
    
    def forward(self, data):
        """Forward pass using pre-computed node features"""
        x, edge_index = data.x, data.edge_index
        
        # Use pre-computed node features directly (no need to embed)
        # x is already [num_nodes, 64] features
        
        # GAT layer
        x = self.gat1(x, edge_index)
        x = F.elu(x)
        x = self.bn1['module'](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # SAGE layer
        x = self.sage(x, edge_index)
        x = F.elu(x)
        x = self.bn2['module'](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        # GCN layer
        x = self.gcn(x, edge_index)
        x = F.elu(x)
        x = self.bn3['module'](x)
        x = F.dropout(x, p=self.dropout, training=self.training)
        
        return x
    
    def predict_rating(self, data, user_idx, book_idx):
        """Predict rating for user-book pair"""
        with torch.no_grad():
            # Ensure indices are integers
            if isinstance(user_idx, torch.Tensor):
                user_idx = user_idx.item()
            if isinstance(book_idx, torch.Tensor):
                book_idx = book_idx.item()
            
            # Validate indices against actual embedding dimensions
            actual_num_users = self.user_embedding.num_embeddings
            actual_num_books = self.book_embedding.num_embeddings
            
            if user_idx >= actual_num_users or user_idx < 0:
                return torch.tensor(2.5)
            if book_idx >= actual_num_books or book_idx < 0:
                return torch.tensor(2.5)
                
            try:
                # Get graph embeddings using forward pass
                graph_embeddings = self.forward(data)
                
                # Get graph embeddings for user and book nodes
                # User nodes are 0 to num_users-1, book nodes are num_users to num_users+num_books-1
                user_graph_emb = graph_embeddings[user_idx]
                book_graph_emb = graph_embeddings[self.num_users + book_idx]
                
                # Get original embeddings (these are used in addition to graph embeddings)
                user_orig_emb = self.user_embedding(torch.tensor(user_idx, dtype=torch.long))
                book_orig_emb = self.book_embedding(torch.tensor(book_idx, dtype=torch.long))
                
                # Get bias terms
                user_bias = self.user_bias(torch.tensor(user_idx, dtype=torch.long))
                book_bias = self.book_bias(torch.tensor(book_idx, dtype=torch.long))
                
                # Combine features for predictor: graph_emb + original_emb
                # user_graph(32) + book_graph(32) + user_orig(64) + book_orig(64) = 192
                combined = torch.cat([
                    user_graph_emb,   # From GCN output
                    book_graph_emb,   # From GCN output  
                    user_orig_emb,    # From embedding layer
                    book_orig_emb     # From embedding layer
                ], dim=0)
                
                # Predict rating
                rating = self.predictor(combined.unsqueeze(0))
                rating = rating + user_bias + book_bias + self.global_bias
                
                return rating.squeeze()
                
            except Exception as e:
                print(f"Prediction error: {e}")
                return torch.tensor(2.5)

class GNNModelLoader:
    """Class to load and use the trained GNN model"""
    
    def __init__(self, timestamp="20250909_234024"):
        self.timestamp = timestamp
        self.model = None
        self.gnn_data = None
        self.user_to_idx = None
        self.book_to_idx = None
        self.idx_to_user = None
        self.idx_to_book = None
        self.books_df = None
        self.comments_df = None
        self.device = torch.device('cpu')
        self.loaded = False
        
    def load_model(self):
        """Load the complete GNN model system"""
        try:
            print(f"🔄 Loading GNN model with timestamp: {self.timestamp}")
            
            # Load model data
            model_path = f"../models/saved_models/gnn_model_{self.timestamp}.pth"
            model_data = torch.load(model_path, map_location=self.device, weights_only=False)
            
            # Load mappings
            with open(f"../models/saved_models/user_to_idx_{self.timestamp}.pkl", 'rb') as f:
                self.user_to_idx = pickle.load(f)
            
            with open(f"../models/saved_models/book_to_idx_{self.timestamp}.pkl", 'rb') as f:
                self.book_to_idx = pickle.load(f)
            
            # Create reverse mappings
            self.idx_to_user = {v: k for k, v in self.user_to_idx.items()}
            self.idx_to_book = {v: k for k, v in self.book_to_idx.items()}
            
            # Load DataFrames
            self.books_df = pickle.load(open(f"../models/saved_models/books_{self.timestamp}.pkl", 'rb'))
            self.comments_df = pickle.load(open(f"../models/saved_models/comments_{self.timestamp}.pkl", 'rb'))
            
            # Load graph data
            try:
                self.gnn_data = torch.load(f"../models/saved_models/gnn_data_{self.timestamp}.pth", 
                                         map_location=self.device, weights_only=False)
            except:
                # Alternative safe loading
                import torch_geometric
                torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
                self.gnn_data = torch.load(f"../models/saved_models/gnn_data_{self.timestamp}.pth", 
                                         map_location=self.device)
            
            # Initialize model with saved model dimensions (not current data dimensions)
            self.model = AdvancedBookRecommenderGNN(
                num_users=12366,    # From saved model
                num_books=1616,     # From saved model
                embedding_dim=64,   # From saved model
                dropout=0.3
            )
            
            # Load trained weights
            self.model.load_state_dict(model_data['model_state_dict'])
            self.model.eval()
            
            self.loaded = True
            print(f"✅ GNN Model loaded successfully!")
            print(f"   • Users: {len(self.user_to_idx):,}")
            print(f"   • Books: {len(self.book_to_idx):,}")
            print(f"   • R² Score: {model_data['test_r2']:.4f}")
            
            return True
            
        except Exception as e:
            print(f"❌ Error loading GNN model: {e}")
            self.loaded = False
            return False
    
    def predict_rating(self, customer_id, product_id):
        """Predict rating for a customer-product pair"""
        if not self.loaded:
            return None, "Model not loaded"
        
        # Check if customer and product exist in mappings
        if customer_id not in self.user_to_idx:
            return None, f"Customer {customer_id} not found in training data"
        
        if product_id not in self.book_to_idx:
            return None, f"Product {product_id} not found in training data"
        
        try:
            user_idx = self.user_to_idx[customer_id]
            book_idx = self.book_to_idx[product_id]  # Don't add offset here
            
            # Validate indices against model dimensions
            if user_idx >= self.model.num_users:
                return None, f"User index {user_idx} exceeds model capacity {self.model.num_users}"
            if book_idx >= self.model.num_books:
                return None, f"Book index {book_idx} exceeds model capacity {self.model.num_books}"
            
            with torch.no_grad():
                rating = self.model.predict_rating(self.gnn_data, user_idx, book_idx)
                return float(rating.item()), None
                
        except Exception as e:
            return None, f"Prediction error: {str(e)}"
    
    def get_user_recommendations(self, customer_id, n_recommendations=5, exclude_rated=True):
        """Get book recommendations for a user - OPTIMIZED VERSION"""
        if not self.loaded:
            return [], "Model not loaded"
        
        if customer_id not in self.user_to_idx:
            return [], f"Customer {customer_id} not found in training data"
        
        try:
            user_idx = self.user_to_idx[customer_id]
            print(f"🔍 Getting recommendations for user {customer_id} (idx: {user_idx})")
            
            # Get books user has already rated (to exclude if requested)
            rated_books = set()
            if exclude_rated:
                user_comments = self.comments_df[self.comments_df['customer_id'] == customer_id]
                rated_books = set(user_comments['product_id'].values)
                print(f"📚 User has rated {len(rated_books)} books, excluding from recommendations")
            
            # Get all books not rated by user - FIX: Use titles since model uses titles as keys
            all_book_titles = set(self.book_to_idx.keys())
            existing_titles = set(self.books_df['title'].unique())
            valid_titles = all_book_titles.intersection(existing_titles)  # Only books with info
            
            # Get rated book titles for this user
            rated_product_ids = set(user_comments['product_id'].values)
            rated_titles = set()
            for pid in rated_product_ids:
                book_info = self.books_df[self.books_df['product_id'] == pid]
                if not book_info.empty:
                    rated_titles.add(book_info.iloc[0]['title'])
            
            candidate_titles = valid_titles - rated_titles
            print(f"🎯 {len(candidate_titles)} candidate books to evaluate (by title matching)")
            print(f"📚 {len(all_book_titles) - len(valid_titles)} books filtered out due to missing book info")
            
            # Validate user index
            actual_num_users = self.model.user_embedding.num_embeddings
            actual_num_books = self.model.book_embedding.num_embeddings
            
            if user_idx >= actual_num_users or user_idx < 0:
                return [], f"Invalid user index: {user_idx}"
            
            # OPTIMIZATION: Compute graph embeddings only once
            print("🧠 Computing graph embeddings...")
            with torch.no_grad():
                graph_embeddings = self.model.forward(self.gnn_data)
                user_graph_emb = graph_embeddings[user_idx]
                user_orig_emb = self.model.user_embedding(torch.tensor(user_idx, dtype=torch.long))
                user_bias = self.model.user_bias(torch.tensor(user_idx, dtype=torch.long))
                
                recommendations = []
                count = 0
                
                print(f"⚡ Processing candidate books...")
                for title in candidate_titles:
                    count += 1
                    if count % 100 == 0:  # Progress update
                        print(f"   Processed {count}/{len(candidate_titles)} books...")
                    
                    book_idx = self.book_to_idx[title]
                    
                    # Validate book index
                    if book_idx >= actual_num_books or book_idx < 0:
                        continue
                    
                    # Fast prediction without calling model.predict_rating
                    try:
                        # Get book embeddings
                        book_graph_emb = graph_embeddings[self.model.num_users + book_idx]
                        book_orig_emb = self.model.book_embedding(torch.tensor(book_idx, dtype=torch.long))
                        book_bias = self.model.book_bias(torch.tensor(book_idx, dtype=torch.long))
                        
                        # Combine features
                        combined = torch.cat([
                            user_graph_emb,   # 32 dims
                            book_graph_emb,   # 32 dims  
                            user_orig_emb,    # 64 dims
                            book_orig_emb     # 64 dims
                        ], dim=0)
                        
                        # Predict rating
                        rating = self.model.predictor(combined.unsqueeze(0))
                        rating = rating + user_bias + book_bias + self.model.global_bias
                        predicted_rating = float(rating.squeeze().item())
                        
                        # Get book information - Now use title to find the product_id
                        book_info = self.books_df[self.books_df['title'] == title].iloc[0]
                        product_id = int(book_info['product_id'])  # Convert to Python int for JSON serialization
                        
                        recommendations.append({
                            'product_id': product_id,
                            'title': str(book_info['title']),  # Ensure string
                            'authors': str(book_info['authors']),  # Ensure string
                            'category': str(book_info.get('category', 'Unknown')),  # Ensure string
                            'cover_link': str(book_info.get('cover_link', '')),  # Ensure string
                            'predicted_rating': float(predicted_rating)  # Ensure float
                        })
                    except Exception as e:
                        print(f"Error predicting for book {title}: {e}")
                        continue
                
                print(f"✅ Generated {len(recommendations)} recommendations")
            
            # Sort by predicted rating and return top N
            recommendations.sort(key=lambda x: x['predicted_rating'], reverse=True)
            return recommendations[:n_recommendations], None
            
        except Exception as e:
            print(f"❌ Recommendation error: {str(e)}")
            import traceback
            traceback.print_exc()
            return [], f"Recommendation error: {str(e)}"
    
    def get_user_history(self, customer_id, limit=10):
        """Get user's reading history"""
        if not self.loaded:
            return [], "Model not loaded"
        
        try:
            user_comments = self.comments_df[self.comments_df['customer_id'] == customer_id]
            user_comments = user_comments.sort_values('rating', ascending=False)
            
            history = []
            for _, comment in user_comments.head(limit).iterrows():
                product_id = int(comment['product_id'])  # Convert to Python int
                book_info = self.books_df[self.books_df['product_id'] == product_id]
                
                if not book_info.empty:
                    book_info = book_info.iloc[0]
                    history.append({
                        'product_id': product_id,
                        'title': str(book_info['title']),  # Ensure string
                        'authors': str(book_info['authors']),  # Ensure string
                        'category': str(book_info.get('category', 'Unknown')),  # Ensure string
                        'cover_link': str(book_info.get('cover_link', '')),  # Ensure string
                        'actual_rating': float(comment['rating'])  # Ensure float
                    })
            
            return history, None
            
        except Exception as e:
            return [], f"History error: {str(e)}"

# Initialize GNN model loader globally
gnn_loader = None
gnn_loaded = False

try:
    # Try to load GNN model
    gnn_loader = GNNModelLoader()
    gnn_loaded = gnn_loader.load_model()
    if gnn_loaded:
        print("✅ GNN Model loaded successfully!")
    else:
        print("❌ GNN Model failed to load")
except Exception as e:
    print(f"❌ Error initializing GNN model: {e}")
    print("⚠️  App will run without GNN functionality")
    gnn_loader = None
    gnn_loaded = False

app = Flask(__name__, template_folder='../templates')

# Configure session
app.secret_key = secrets.token_hex(16)
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SECURE'] = False  # Set to True in production with HTTPS

# Helper function to check GNN availability
def is_gnn_available():
    """Check if GNN model is loaded and available"""
    return gnn_loaded and gnn_loader is not None

def ensure_gnn_available():
    """Return error response if GNN is not available"""
    if not is_gnn_available():
        return jsonify({
            'status': 'error',
            'message': 'GNN model not loaded. Please check model files.'
        }), 503
    return None

# Helper functions for session-based tracking
def track_book_view(book_info):
    """Lưu thông tin sách đã xem vào session"""
    if 'viewed_books' not in session:
        session['viewed_books'] = []
    
    # Tránh duplicate - check by book title
    viewed_titles = [book['title'] for book in session['viewed_books']]
    if book_info['title'] not in viewed_titles:
        session['viewed_books'].append({
            'id': book_info.get('id', book_info['title']),
            'title': book_info['title'],
            'category': book_info.get('category', ''),
            'author': book_info.get('author', ''),
            'timestamp': datetime.now().isoformat()
        })
        # Giới hạn chỉ lưu 20 sách gần nhất
        session['viewed_books'] = session['viewed_books'][-20:]
        session.permanent = True

def analyze_user_preferences():
    """Phân tích sở thích của user từ lịch sử xem"""
    if 'viewed_books' not in session or not session['viewed_books']:
        return None
    
    # Đếm số lần xem theo thể loại
    category_count = {}
    for book in session['viewed_books']:
        category = book.get('category', 'Unknown')
        if category and category != 'Unknown':
            category_count[category] = category_count.get(category, 0) + 1
    
    if not category_count:
        return None
    
    # Tìm thể loại ưa thích nhất
    favorite_category = max(category_count, key=category_count.get)
    
    return {
        'favorite_category': favorite_category,
        'category_counts': category_count,
        'total_viewed': len(session['viewed_books'])
    }

def get_books_by_category(category, limit=8, exclude_titles=None):
    """Lấy sách theo thể loại từ books DataFrame"""
    if exclude_titles is None:
        exclude_titles = []
    
    try:
        # Filter books by category, exclude already viewed
        filtered_books = books[
            (books['category'] == category) & 
            (~books['title'].isin(exclude_titles))
        ]
        
        # Sort by number of ratings (popularity) and get top books
        if 'n_review' in filtered_books.columns:
            top_books = filtered_books.nlargest(limit, 'n_review')
        else:
            top_books = filtered_books.head(limit)
        
        # Convert to list of dictionaries
        result = []
        for idx, row in top_books.iterrows():
            book_data = {
                'title': row['title'],
                'author': row['authors'],
                'category': row.get('category', ''),
                'cover_url': row.get('cover_link', ''),
                'rating': 0,  # Không có rating trong data này
                'num_ratings': row.get('n_review', 0)
            }
            result.append(book_data)
        
        return result
    except Exception as e:
        print(f"Error getting books by category: {e}")
        return []

@app.route('/')
def index():
    # Debug: Print column names to check what's available
    print("Popular DataFrame columns:", popular_df.columns.tolist())
    
    return render_template('index.html',
                           book_name = list(popular_df['book_title'].values),
                           author=list(popular_df['authors'].values),
                           image=list(popular_df['cover_link'].values),
                           votes=list(popular_df['num_ratings'].values),
                           rating=list(popular_df['avg_rating'].values)
                           )

@app.route('/recommend')
def recommend_ui():
    return render_template('recommend_new.html')

@app.route('/user_recommend')
def user_recommend_ui():
    # Get list of available users for the dropdown
    available_users = sorted(ml_data_filtered['customer_id'].unique())
    return render_template('user_recommend.html', users=available_users)

@app.route('/gnn_recommend')
def gnn_recommend_ui():
    """GNN recommendation interface"""
    if not gnn_loaded or gnn_loader is None:
        return render_template('gnn_recommend.html', 
                             error="GNN model not loaded. Please check model files.",
                             users=[])
    
    # Get list of available users from GNN model
    available_users = sorted(list(gnn_loader.user_to_idx.keys()))
    return render_template('gnn_recommend.html', users=available_users)

@app.route('/user_recommend_books', methods=['POST'])
def user_recommend():
    user_id = request.form.get('user_id')
    algorithm = request.form.get('algorithm', 'both')  # 'gnn', 'ml', or 'both'
    
    try:
        user_id = int(user_id)
        
        # Check if user exists in ML data
        ml_available = user_id in ml_data_filtered['customer_id'].values
        
        # Check if user exists in GNN data
        gnn_available = gnn_loaded and gnn_loader is not None and user_id in gnn_loader.user_to_idx
        
        if not ml_available and not gnn_available:
            return render_template('user_recommend.html', 
                                 users=sorted(ml_data_filtered['customer_id'].unique()),
                                 error=f"User ID {user_id} không có trong hệ thống nào.")
        
        # Get user history
        user_history = []
        if ml_available:
            user_history_ml = ml_data_filtered[ml_data_filtered['customer_id'] == user_id].sort_values('rating', ascending=False)
            for _, book in user_history_ml.head(5).iterrows():
                user_history.append({
                    'title': book['book_title'],
                    'rating': book['rating']
                })
        elif gnn_available:
            user_history_gnn, _ = gnn_loader.get_user_history(user_id, limit=5)
            for book in user_history_gnn:
                user_history.append({
                    'title': book['title'],
                    'rating': book['actual_rating']
                })
        
        # Get recommendations based on selected algorithm
        ml_recommendations = []
        gnn_recommendations = []
        
        if algorithm in ['ml', 'both'] and ml_available:
            ml_recommendations = ml_recommend_for_user(user_id, n_recommendations=5)
        
        if algorithm in ['gnn', 'both'] and gnn_available and gnn_loader is not None:
            gnn_recs, gnn_error = gnn_loader.get_user_recommendations(user_id, n_recommendations=5)
            if not gnn_error:
                gnn_recommendations = gnn_recs
        
        return render_template('user_recommend.html', 
                             users=sorted(ml_data_filtered['customer_id'].unique()),
                             ml_recommendations=ml_recommendations,
                             gnn_recommendations=gnn_recommendations,
                             user_history=user_history,
                             user_id=user_id,
                             algorithm=algorithm,
                             ml_available=ml_available,
                             gnn_available=gnn_available)
        
    except ValueError:
        return render_template('user_recommend.html',
                             users=sorted(ml_data_filtered['customer_id'].unique()),
                             error="Vui lòng nhập User ID hợp lệ.")
    except Exception as e:
        return render_template('user_recommend.html',
                             users=sorted(ml_data_filtered['customer_id'].unique()),
                             error=f"Có lỗi xảy ra: {str(e)}")

@app.route('/gnn_recommend_books', methods=['POST'])
def gnn_recommend():
    """Handle GNN recommendation form submission"""
    if not gnn_loaded:
        available_users = []
        return render_template('gnn_recommend.html', 
                             users=available_users,
                             error="GNN model not loaded. Please check model files.")
    
    available_users = sorted(list(gnn_loader.user_to_idx.keys()))
    
    try:
        customer_id = int(request.form.get('customer_id'))
        limit = int(request.form.get('limit', 5))
        exclude_rated = request.form.get('exclude_rated') == 'on'
        
        # Get GNN recommendations
        recommendations, error = gnn_loader.get_user_recommendations(
            customer_id, n_recommendations=limit, exclude_rated=exclude_rated
        )
        
        if error:
            return render_template('gnn_recommend.html', 
                                 users=available_users,
                                 error=error,
                                 customer_id=customer_id)
        
        # Get user history
        history, _ = gnn_loader.get_user_history(customer_id, limit=10)
        
        # Get comparison with ML recommendations if available
        ml_recommendations = []
        if customer_id in ml_data_filtered['customer_id'].values:
            try:
                ml_recommendations = ml_recommend_for_user(customer_id, limit)
            except:
                pass
        
        return render_template('gnn_recommend.html', 
                             users=available_users,
                             customer_id=customer_id,
                             gnn_recommendations=recommendations,
                             ml_recommendations=ml_recommendations,
                             user_history=history,
                             limit=limit,
                             exclude_rated=exclude_rated)
        
    except ValueError:
        return render_template('gnn_recommend.html',
                             users=available_users,
                             error="Vui lòng nhập Customer ID hợp lệ.")
    except Exception as e:
        return render_template('gnn_recommend.html',
                             users=available_users,
                             error=f"Có lỗi xảy ra: {str(e)}")

def ml_recommend_for_user(user_id, n_recommendations=5):
    """
    Gợi ý sách cho user dựa trên Matrix Factorization (SVD)
    """
    # Lấy tất cả sách trong hệ thống
    all_books = ml_data_filtered['book_title'].unique()
    
    # Lấy sách mà user đã đánh giá
    user_rated_books = ml_data_filtered[ml_data_filtered['customer_id'] == user_id]['book_title'].unique()
    
    # Tìm sách chưa được đánh giá
    unrated_books = [book for book in all_books if book not in user_rated_books]
    
    # Dự đoán rating cho các sách chưa đánh giá
    predictions = []
    for book in unrated_books:
        pred = svd_model.predict(user_id, book)
        predictions.append((book, pred.est))
    
    # Sắp xếp theo predicted rating giảm dần
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Lấy top N recommendations
    top_recommendations = predictions[:n_recommendations]
    
    # Format kết quả
    recommendations = []
    for book_title, predicted_rating in top_recommendations:
        book_info = books[books['title'] == book_title]
        if not book_info.empty:
            book_info = book_info.iloc[0]
            recommendations.append({
                'title': book_title,
                'author': book_info['authors'],
                'predicted_rating': round(predicted_rating, 2),
                'cover_link': book_info['cover_link']
            })
    
    return recommendations

def recommend_collaborative(book_name):
    """
    Pure collaborative filtering recommendation
    """
    if book_name not in pt.index:
        return []
    
    index = np.where(pt.index == book_name)[0][0]
    similar_items = sorted(list(enumerate(similarity_scores[index])), key=lambda x: x[1], reverse=True)[1:5]

    data = []
    for i in similar_items:
        item = []
        temp_df = books[books['title'] == pt.index[i[0]]]
        if not temp_df.empty:
            item.extend(list(temp_df.drop_duplicates('title')['title'].values))
            item.extend(list(temp_df.drop_duplicates('title')['authors'].values))
            item.extend(list(temp_df.drop_duplicates('title')['cover_link'].values))
            data.append(item)
    
    return data

def content_based_recommend(book_name):
    """
    Content-based recommendation using multiple features:
    1. Same category (thể loại)
    2. Same author 
    3. Books liked by users who rated the target book highly
    """
    # Find the book in books dataset
    book_info = books[books['title'] == book_name]
    if book_info.empty:
        print(f"Book '{book_name}' not found in database.")
        return []
    
    book_author = book_info.iloc[0]['authors']
    book_category = book_info.iloc[0].get('category', '')
    print(f"📚 Content-based recommendation for '{book_name}'")
    print(f"   Author: {book_author}")
    print(f"   Category: {book_category}")
    
    recommendations = []
    
    # Strategy 1: Same category books (PRIMARY)
    if book_category and book_category != '' and book_category != 'Others':
        same_category_books = books[
            (books['category'] == book_category) & 
            (books['title'] != book_name)
        ]
        
        if len(same_category_books) > 0:
            print(f"✅ Found {len(same_category_books)} books in same category: '{book_category}'")
            # Lấy 4 sách cùng thể loại, ưu tiên sách có nhiều review
            if 'n_review' in same_category_books.columns:
                top_category_books = same_category_books.sort_values('n_review', ascending=False).head(4)
            else:
                top_category_books = same_category_books.head(4)

            for _, book in top_category_books.iterrows():
                recommendations.append([
                    book['title'],
                    book['authors'], 
                    book['cover_link']
                ])
        else:
            print(f"❌ No books found in category: '{book_category}'")
    else:
        print(f"⚠️  Category is empty or 'Others', using author instead")
    
    # Strategy 2: Same author books (if category didn't provide enough)
    if len(recommendations) < 2:
        same_author_books = books[
            (books['authors'] == book_author) & 
            (books['title'] != book_name) &
            (~books['title'].isin([rec[0] for rec in recommendations]))  # Avoid duplicates
        ]
        
        if len(same_author_books) > 0:
            needed = 2 - len(recommendations)
            print(f"✅ Adding {min(needed, len(same_author_books))} books by same author")
            for _, book in same_author_books.head(needed).iterrows():
                recommendations.append([
                    book['title'],
                    book['authors'], 
                    book['cover_link']
                ])
    
    # Strategy 3: Find users who rated this book highly and see what else they liked
    if len(recommendations) < 4 and book_name in ml_data_filtered['book_title'].values:
        users_who_liked = ml_data_filtered[
            (ml_data_filtered['book_title'] == book_name) & 
            (ml_data_filtered['rating'] >= 4)
        ]['customer_id'].unique()
        
        if len(users_who_liked) > 0:
            print(f"✅ Found {len(users_who_liked)} users who liked this book")
            
            # Get other highly rated books by these users
            other_books = ml_data_filtered[
                (ml_data_filtered['customer_id'].isin(users_who_liked)) & 
                (ml_data_filtered['book_title'] != book_name) & 
                (ml_data_filtered['rating'] >= 4)
            ]
            
            # Count popularity among these users
            book_popularity = other_books.groupby('book_title').size().sort_values(ascending=False)
            
            # Add top books from similar taste users
            needed = 4 - len(recommendations)
            added_count = 0
            for similar_book in book_popularity.index:
                if added_count >= needed:
                    break
                    
                # Check if not already in recommendations
                if not any(rec[0] == similar_book for rec in recommendations):
                    book_details = books[books['title'] == similar_book]
                    if not book_details.empty:
                        book_details = book_details.iloc[0]
                        recommendations.append([
                            book_details['title'],
                            book_details['authors'],
                            book_details['cover_link']
                        ])
                        added_count += 1
            
            if added_count > 0:
                print(f"✅ Added {added_count} books from users with similar taste")
    
    # Strategy 4: If still not enough, use popular books from same category as fallback
    while len(recommendations) < 4:
        # Try same category popular books first
        if book_category and book_category != '' and book_category != 'Others':
            category_popular = popular_df[
                popular_df['book_title'].isin(
                    books[books['category'] == book_category]['title']
                ) &
                (~popular_df['book_title'].isin([rec[0] for rec in recommendations]))
            ]
            
            if not category_popular.empty:
                next_book = category_popular.iloc[0]
                recommendations.append([
                    next_book['book_title'],
                    next_book['authors'],
                    next_book['cover_link']
                ])
                print(f"✅ Added popular book from same category")
                continue
        
        # Final fallback: general popular books
        remaining_popular = popular_df[
            ~popular_df['book_title'].isin([rec[0] for rec in recommendations])
        ]
        
        if remaining_popular.empty:
            break
            
        next_popular = remaining_popular.iloc[0]
        recommendations.append([
            next_popular['book_title'],
            next_popular['authors'],
            next_popular['cover_link']
        ])
        print(f"✅ Added general popular book as fallback")
    
    print(f"📊 Content-based recommendations: {len(recommendations)} books found")
    return recommendations

def hybrid_recommend(book_name):
    """
    Advanced Hybrid recommendation system:
    - Use collaborative filtering if book is in pivot table
    - Use content-based filtering if not in pivot table
    """
    
    # Case 1: Book exists in collaborative filtering system
    if book_name in pt.index:
        print(f"✅ Using Collaborative Filtering for '{book_name}'")
        return recommend_collaborative(book_name)
    
    # Case 2: Book not in pivot table - use content-based approach
    print(f"🔄 Book '{book_name}' not in collaborative system. Using Content-Based approach...")
    return content_based_recommend(book_name)

@app.route('/recommend_books',methods=['post'])
def recommend_books():
    user_input = request.form.get('user_input')
    
    # Get information about the current book being searched
    current_book_info = None
    book_details = books[books['title'].str.contains(user_input, case=False, na=False)]
    
    if not book_details.empty:
        # Get the best match (exact match first, then partial match)
        exact_match = book_details[book_details['title'].str.lower() == user_input.lower()]
        if not exact_match.empty:
            book_match = exact_match.iloc[0]
        else:
            book_match = book_details.iloc[0]
            
        current_book_info = {
            'title': book_match['title'],
            'author': book_match['authors'],
            'category': book_match.get('category', 'N/A'),
            'cover_link': book_match['cover_link'],
            'pages': book_match.get('pages', 'N/A'),
            'n_review': book_match.get('n_review', 0)
        }
    
    # Determine which algorithm will be used
    algorithm_used = "Collaborative Filtering" if user_input in pt.index else "Content-Based Filtering"
    
    # Use hybrid recommendation
    data = hybrid_recommend(user_input)
    
    if not data:
        return render_template('recommend_new.html', 
                             data=[], 
                             current_book=current_book_info,
                             search_query=user_input,
                             algorithm_used=algorithm_used,
                             error=f"Không tìm thấy gợi ý cho sách '{user_input}'. Vui lòng thử tên sách khác.")

    print(f"Hybrid recommendations for '{user_input}':", len(data), "books found")
    return render_template('recommend_new.html', 
                         data=data, 
                         current_book=current_book_info,
                         search_query=user_input,
                         algorithm_used=algorithm_used)

def ml_recommend_for_user(user_id, n_recommendations=5):
    """
    Gợi ý sách cho user dựa trên Matrix Factorization (SVD)
    """
    # Lấy tất cả sách trong hệ thống
    all_books = ml_data_filtered['book_title'].unique()
    
    # Lấy sách mà user đã đánh giá
    user_rated_books = ml_data_filtered[ml_data_filtered['customer_id'] == user_id]['book_title'].unique()
    
    # Tìm sách chưa được đánh giá
    unrated_books = [book for book in all_books if book not in user_rated_books]
    
    # Dự đoán rating cho các sách chưa đánh giá
    predictions = []
    for book in unrated_books:
        pred = svd_model.predict(user_id, book)
        predictions.append((book, pred.est))
    
    # Sắp xếp theo predicted rating giảm dần
    predictions.sort(key=lambda x: x[1], reverse=True)
    
    # Lấy top N recommendations
    top_recommendations = predictions[:n_recommendations]
    
    # Format kết quả
    recommendations = []
    for book_title, predicted_rating in top_recommendations:
        book_info = books[books['title'] == book_title]
        if not book_info.empty:
            book_info = book_info.iloc[0]
            recommendations.append({
                'title': book_title,
                'author': book_info['authors'],
                'predicted_rating': round(predicted_rating, 2),
                'cover_link': book_info['cover_link']
            })
    
    return recommendations

# ======================= API ENDPOINTS FOR POSTMAN TESTING =======================

@app.route('/api/books/popular', methods=['GET'])
def api_popular_books():
    """
    API to get popular books
    GET /api/books/popular
    """
    try:
        books_data = []
        for i in range(len(popular_df)):
            book = {
                'title': popular_df.iloc[i]['book_title'],
                'author': popular_df.iloc[i]['authors'],
                'image_url': popular_df.iloc[i]['cover_link'],
                'num_ratings': int(popular_df.iloc[i]['num_ratings']),
                'avg_rating': float(popular_df.iloc[i]['avg_rating'])
            }
            books_data.append(book)
        
        return jsonify({
            'status': 'success',
            'data': books_data,
            'total_books': len(books_data)
        }), 200
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/books/recommend', methods=['POST'])
def api_recommend_books():
    """
    API to get book recommendations
    POST /api/books/recommend
    Body: {"book_title": "Book Name"}
    """
    try:
        data = request.get_json()
        if not data or 'book_title' not in data:
            return jsonify({
                'status': 'error',
                'message': 'book_title is required in request body'
            }), 400
        
        book_title = data['book_title']
        
        # Get recommendations using hybrid approach
        recommendations = hybrid_recommend(book_title)
        
        if not recommendations:
            return jsonify({
                'status': 'error',
                'message': f'No recommendations found for "{book_title}"'
            }), 404
        
        return jsonify({
            'status': 'success',
            'search_query': book_title,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/books/collaborative', methods=['POST'])
def api_collaborative_recommend():
    """
    API for collaborative filtering recommendations
    POST /api/books/collaborative
    Body: {"book_title": "Book Name"}
    """
    try:
        data = request.get_json()
        if not data or 'book_title' not in data:
            return jsonify({
                'status': 'error',
                'message': 'book_title is required in request body'
            }), 400
        
        book_title = data['book_title']
        
        # Get collaborative recommendations
        recommendations = recommend_collaborative(book_title)
        
        return jsonify({
            'status': 'success',
            'algorithm': 'Collaborative Filtering',
            'search_query': book_title,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/books/content-based', methods=['POST'])
def api_content_based_recommend():
    """
    API for content-based filtering recommendations
    POST /api/books/content-based
    Body: {"book_title": "Book Name"}
    """
    try:
        data = request.get_json()
        if not data or 'book_title' not in data:
            return jsonify({
                'status': 'error',
                'message': 'book_title is required in request body'
            }), 400
        
        book_title = data['book_title']
        
        # Get content-based recommendations
        recommendations = content_based_recommend(book_title)
        
        return jsonify({
            'status': 'success',
            'algorithm': 'Content-Based Filtering',
            'search_query': book_title,
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/users/<int:user_id>/recommendations', methods=['GET'])
def api_user_recommendations(user_id):
    """
    API to get ML-based user recommendations
    GET /api/users/{user_id}/recommendations
    """
    try:
        # Check if user exists
        if user_id not in ml_data_filtered['customer_id'].values:
            return jsonify({
                'status': 'error',
                'message': f'User {user_id} not found in database'
            }), 404
        
        # Get ML recommendations
        recommendations = ml_recommend_for_user(user_id)
        
        return jsonify({
            'status': 'success',
            'user_id': user_id,
            'algorithm': 'Matrix Factorization (SVD)',
            'recommendations': recommendations,
            'total_recommendations': len(recommendations)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/books/search/<string:query>', methods=['GET'])
def api_search_books(query):
    """
    API to search books by title
    GET /api/books/search/{query}
    """
    try:
        # Search for books containing the query string
        matching_books = books[books['title'].str.contains(query, case=False, na=False)]
        
        if matching_books.empty:
            return jsonify({
                'status': 'error',
                'message': f'No books found matching "{query}"'
            }), 404
        
        books_data = []
        for _, book in matching_books.head(10).iterrows():  # Limit to 10 results
            book_info = {
                'title': book['title'],
                'author': book['authors'],
                'category': book['category'] if 'category' in book else 'N/A',
                'pages': book['pages'] if 'pages' in book else 'N/A',
                'cover_link': book['cover_link']
            }
            books_data.append(book_info)
        
        return jsonify({
            'status': 'success',
            'search_query': query,
            'books': books_data,
            'total_found': len(matching_books),
            'showing': len(books_data)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/users', methods=['GET'])
def api_get_users():
    """
    API to get list of available users
    GET /api/users
    """
    try:
        available_users = sorted(ml_data_filtered['customer_id'].unique().tolist())
        
        return jsonify({
            'status': 'success',
            'users': available_users[:50],  # Limit to first 50 users
            'total_users': len(available_users)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/health', methods=['GET'])
def api_health_check():
    """
    API health check
    GET /api/health
    """
    return jsonify({
        'status': 'healthy',
        'service': 'Book Recommender System',
        'version': '1.0',
        'models_loaded': {
            'popular_books': len(popular_df) > 0,
            'collaborative_filtering': similarity_scores is not None,
            'ml_model': svd_model is not None,
            'books_database': len(books) > 0
        }
    }), 200

# ===== SESSION-BASED TRACKING APIs =====

@app.route('/api/track-view', methods=['POST'])
def api_track_view():
    """
    Track when user views a book
    POST /api/track-view
    Body: {"title": "Book Title", "author": "Author", "category": "Category"}
    """
    try:
        data = request.get_json()
        
        if not data or 'title' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Book title is required'
            }), 400
        
        # Track the book view
        book_info = {
            'title': data['title'],
            'author': data.get('author', ''),
            'category': data.get('category', ''),
            'id': data.get('id', data['title'])
        }
        
        track_book_view(book_info)
        
        return jsonify({
            'status': 'success',
            'message': 'Book view tracked successfully',
            'total_viewed': len(session.get('viewed_books', []))
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/personalized', methods=['GET'])
def api_get_personalized_recommendations():
    """
    Get personalized recommendations based on viewing history
    GET /api/personalized?limit=8
    """
    try:
        limit = int(request.args.get('limit', 8))
        
        # Analyze user preferences
        preferences = analyze_user_preferences()
        
        if not preferences:
            return jsonify({
                'status': 'success',
                'has_history': False,
                'message': 'No viewing history found'
            }), 200
        
        # Get viewed book titles to exclude
        viewed_titles = [book['title'] for book in session.get('viewed_books', [])]
        
        # Get recommendations from favorite category
        favorite_category = preferences['favorite_category']
        recommended_books = get_books_by_category(
            category=favorite_category,
            limit=limit,
            exclude_titles=viewed_titles
        )
        
        return jsonify({
            'status': 'success',
            'has_history': True,
            'favorite_category': favorite_category,
            'total_viewed': preferences['total_viewed'],
            'category_breakdown': preferences['category_counts'],
            'recommendations': recommended_books,
            'reason': f"Dựa trên sở thích {favorite_category} của bạn"
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/recommendations', methods=['GET'])
def api_get_recommendations():
    """
    Get popular book recommendations
    GET /api/recommendations?type=popular&limit=8
    """
    try:
        rec_type = request.args.get('type', 'popular')
        limit = int(request.args.get('limit', 8))
        
        if rec_type == 'popular':
            # Get popular books
            popular_books = popular_df.head(limit)
            
            recommendations = []
            for idx, row in popular_books.iterrows():
                book_data = {
                    'title': row['book_title'],
                    'author': row['authors'],
                    'cover_url': row['cover_link'],
                    'rating': row['avg_rating'],
                    'num_ratings': row['num_ratings'],
                    'category': 'Popular'  # Popular books may not have specific category
                }
                recommendations.append(book_data)
            
            return jsonify({
                'status': 'success',
                'type': 'popular',
                'recommendations': recommendations,
                'total': len(recommendations)
            }), 200
        
        else:
            return jsonify({
                'status': 'error',
                'message': f'Unsupported recommendation type: {rec_type}'
            }), 400
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/user-history', methods=['GET'])
def api_get_user_history():
    """
    Get user's viewing history
    GET /api/user-history
    """
    try:
        viewed_books = session.get('viewed_books', [])
        
        return jsonify({
            'status': 'success',
            'total_viewed': len(viewed_books),
            'history': viewed_books
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/clear-history', methods=['DELETE'])
def api_clear_history():
    """
    Clear user's viewing history
    DELETE /api/clear-history
    """
    try:
        if 'viewed_books' in session:
            del session['viewed_books']
        
        return jsonify({
            'status': 'success',
            'message': 'Viewing history cleared successfully'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/categories', methods=['GET'])
def api_get_categories():
    """
    Get all available book categories
    GET /api/categories
    """
    try:
        categories = books['Category'].value_counts().to_dict()
        
        return jsonify({
            'status': 'success',
            'categories': categories,
            'total_categories': len(categories)
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error', 
            'message': str(e)
        }), 500

# ======================= GNN API ENDPOINTS =======================

@app.route('/api/gnn/health', methods=['GET'])
def api_gnn_health():
    """
    Check GNN model health status
    GET /api/gnn/health
    """
    return jsonify({
        'status': 'healthy' if gnn_loaded and gnn_loader is not None else 'error',
        'model_loaded': gnn_loaded and gnn_loader is not None,
        'users_available': len(gnn_loader.user_to_idx) if gnn_loaded and gnn_loader is not None else 0,
        'books_available': len(gnn_loader.book_to_idx) if gnn_loaded and gnn_loader is not None else 0,
        'timestamp': gnn_loader.timestamp if gnn_loaded and gnn_loader is not None else None
    }), 200 if gnn_loaded and gnn_loader is not None else 503

@app.route('/api/gnn/users/<int:customer_id>/recommendations', methods=['GET'])
def api_gnn_user_recommendations(customer_id):
    """
    Get GNN-based recommendations for a specific user
    GET /api/gnn/users/{customer_id}/recommendations?limit=5&exclude_rated=true
    """
    try:
        # Check if GNN is available
        error_response = ensure_gnn_available()
        if error_response:
            return error_response
        
        # Get query parameters
        limit = int(request.args.get('limit', 5))
        exclude_rated = request.args.get('exclude_rated', 'true').lower() == 'true'
        
        # Get recommendations
        recommendations, error = gnn_loader.get_user_recommendations(
            customer_id, n_recommendations=limit, exclude_rated=exclude_rated
        )
        
        if error:
            return jsonify({
                'status': 'error',
                'message': error
            }), 400
        
        # Get user history for context
        history, _ = gnn_loader.get_user_history(customer_id, limit=5)
        
        return jsonify({
            'status': 'success',
            'customer_id': customer_id,
            'algorithm': 'Graph Neural Network (GNN)',
            'recommendations': recommendations,
            'user_history': history,
            'total_recommendations': len(recommendations),
            'parameters': {
                'limit': limit,
                'exclude_rated': exclude_rated
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/gnn/predict', methods=['POST'])
def api_gnn_predict_rating():
    """
    Predict rating for a specific user-book pair using GNN
    POST /api/gnn/predict
    Body: {"customer_id": 123, "product_id": "B000ABC"}
    """
    try:
        if not gnn_loaded:
            return jsonify({
                'status': 'error',
                'message': 'GNN model not loaded'
            }), 503
        
        data = request.get_json()
        if not data or 'customer_id' not in data or 'product_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing customer_id or product_id in request body'
            }), 400
        
        customer_id = data['customer_id']
        product_id = data['product_id']
        
        # Predict rating
        predicted_rating, error = gnn_loader.predict_rating(customer_id, product_id)
        
        if error:
            return jsonify({
                'status': 'error',
                'message': error
            }), 400
        
        # Get book information
        book_info = gnn_loader.books_df[gnn_loader.books_df['product_id'] == product_id]
        book_details = {}
        if not book_info.empty:
            book = book_info.iloc[0]
            book_details = {
                'title': book['title'],
                'authors': book['authors'],
                'category': book.get('category', 'Unknown'),
                'cover_link': book.get('cover_link', '')
            }
        
        # Check if user actually rated this book
        actual_rating = None
        user_comment = gnn_loader.comments_df[
            (gnn_loader.comments_df['customer_id'] == customer_id) & 
            (gnn_loader.comments_df['product_id'] == product_id)
        ]
        if not user_comment.empty:
            actual_rating = float(user_comment.iloc[0]['rating'])
        
        return jsonify({
            'status': 'success',
            'customer_id': customer_id,
            'product_id': product_id,
            'book_details': book_details,
            'predicted_rating': round(predicted_rating, 2),
            'actual_rating': actual_rating,
            'algorithm': 'Graph Neural Network (GNN)'
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/gnn/users', methods=['GET'])
def api_gnn_get_users():
    """
    Get list of users available in GNN model
    GET /api/gnn/users?limit=100&offset=0
    """
    try:
        if not gnn_loaded:
            return jsonify({
                'status': 'error',
                'message': 'GNN model not loaded'
            }), 503
        
        limit = int(request.args.get('limit', 100))
        offset = int(request.args.get('offset', 0))
        
        all_users = sorted(list(gnn_loader.user_to_idx.keys()))
        total_users = len(all_users)
        
        # Pagination
        start = offset
        end = min(offset + limit, total_users)
        users_page = all_users[start:end]
        
        # Get some stats for each user
        users_with_stats = []
        for user_id in users_page:
            user_comments = gnn_loader.comments_df[gnn_loader.comments_df['customer_id'] == user_id]
            users_with_stats.append({
                'customer_id': user_id,
                'total_ratings': len(user_comments),
                'avg_rating': float(user_comments['rating'].mean()) if len(user_comments) > 0 else 0,
                'favorite_books': user_comments.nlargest(3, 'rating')['product_id'].tolist() if len(user_comments) > 0 else []
            })
        
        return jsonify({
            'status': 'success',
            'users': users_with_stats,
            'pagination': {
                'total_users': total_users,
                'offset': offset,
                'limit': limit,
                'returned': len(users_with_stats)
            }
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/gnn/users/active', methods=['GET'])
def api_gnn_get_active_users():
    """
    Get list of most active users (sorted by rating count)
    GET /api/gnn/users/active?limit=20&min_ratings=5
    """
    try:
        if not gnn_loaded:
            return jsonify({
                'status': 'error',
                'message': 'GNN model not loaded'
            }), 503
        
        limit = int(request.args.get('limit', 20))
        min_ratings = int(request.args.get('min_ratings', 5))
        
        # Calculate user statistics
        user_stats = gnn_loader.comments_df.groupby('customer_id').agg({
            'rating': ['count', 'mean'],
            'product_id': 'nunique'
        }).round(2)
        
        user_stats.columns = ['rating_count', 'avg_rating', 'unique_books']
        user_stats = user_stats.reset_index()
        
        # Filter by minimum ratings
        active_users = user_stats[user_stats['rating_count'] >= min_ratings]
        active_users = active_users.sort_values('rating_count', ascending=False)
        
        # Get top users
        top_users = active_users.head(limit)
        
        # Format response
        users_list = []
        for _, user in top_users.iterrows():
            user_comments = gnn_loader.comments_df[gnn_loader.comments_df['customer_id'] == user['customer_id']]
            
            # Get favorite books (highest rated)
            favorite_books = user_comments.nlargest(3, 'rating')[['product_id', 'rating']].to_dict('records')
            
            # Get recent activity
            recent_books = user_comments.tail(3)[['product_id', 'rating']].to_dict('records')
            
            users_list.append({
                'customer_id': int(user['customer_id']),
                'statistics': {
                    'total_ratings': int(user['rating_count']),
                    'avg_rating': float(user['avg_rating']),
                    'unique_books_rated': int(user['unique_books'])
                },
                'favorite_books': favorite_books,
                'recent_activity': recent_books,
                'recommendation_ready': True  # These users have enough data for good recommendations
            })
        
        # Overall statistics
        total_users_with_min_ratings = len(active_users)
        overall_stats = {
            'total_users_in_model': len(gnn_loader.user_to_idx),
            'users_with_min_ratings': total_users_with_min_ratings,
            'percentage_active': round((total_users_with_min_ratings / len(gnn_loader.user_to_idx)) * 100, 2),
            'filter_criteria': {
                'min_ratings': min_ratings,
                'returned_count': len(users_list)
            }
        }
        
        return jsonify({
            'status': 'success',
            'active_users': users_list,
            'statistics': overall_stats,
            'suggested_test_users': [user['customer_id'] for user in users_list[:5]]  # Top 5 for testing
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/gnn/compare', methods=['POST'])
def api_gnn_compare_algorithms():
    """
    Compare GNN recommendations with other algorithms for a user
    POST /api/gnn/compare
    Body: {"customer_id": 123, "limit": 5}
    """
    try:
        if not gnn_loaded:
            return jsonify({
                'status': 'error',
                'message': 'GNN model not loaded'
            }), 503
        
        data = request.get_json()
        if not data or 'customer_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing customer_id in request body'
            }), 400
        
        customer_id = data['customer_id']
        limit = data.get('limit', 5)
        
        # Get GNN recommendations
        gnn_recs, gnn_error = gnn_loader.get_user_recommendations(customer_id, limit)
        
        if gnn_error:
            return jsonify({
                'status': 'error',
                'message': f'GNN error: {gnn_error}'
            }), 400
        
        # Get ML (SVD) recommendations if user exists in ML data
        ml_recs = []
        if customer_id in ml_data_filtered['customer_id'].values:
            try:
                ml_recs = ml_recommend_for_user(customer_id, limit)
            except:
                ml_recs = []
        
        # Get user history
        history, _ = gnn_loader.get_user_history(customer_id, limit=5)
        
        return jsonify({
            'status': 'success',
            'customer_id': customer_id,
            'user_history': history,
            'recommendations': {
                'gnn': {
                    'algorithm': 'Graph Neural Network',
                    'recommendations': gnn_recs,
                    'count': len(gnn_recs)
                },
                'ml_svd': {
                    'algorithm': 'Matrix Factorization (SVD)',
                    'recommendations': ml_recs,
                    'count': len(ml_recs)
                }
            },
            'comparison_notes': [
                "GNN uses graph structure and node relationships",
                "SVD uses matrix factorization on user-item ratings",
                "GNN may provide more diverse recommendations",
                "SVD is faster for large-scale predictions"
            ]
        }), 200
        
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 500

@app.route('/api/gnn/recommend', methods=['POST'])
def api_gnn_simple_recommend():
    """
    Simple API to get book recommendations for a user using GNN
    POST /api/gnn/recommend
    Body: {"customer_id": 123, "limit": 5}
    """
    import time
    start_time = time.time()
    
    try:
        print(f"📨 Received recommendation request at {time.strftime('%H:%M:%S')}")
        
        if not gnn_loaded:
            return jsonify({
                'status': 'error',
                'message': 'GNN model not available. Please check model loading.',
                'fallback': 'Try using /api/recommend for ML-based recommendations'
            }), 503
        
        data = request.get_json()
        if not data or 'customer_id' not in data:
            return jsonify({
                'status': 'error',
                'message': 'Missing customer_id in request body',
                'example': {'customer_id': 123, 'limit': 5}
            }), 400
        
        customer_id = data['customer_id']
        limit = data.get('limit', 10)
        
        print(f"🎯 Processing request for user {customer_id}, limit: {limit}")
        
        # Validate limit
        if limit > 50:
            limit = 50
        elif limit < 1:
            limit = 1
        
        # Get recommendations with timeout protection
        print(f"⚡ Starting recommendation generation...")
        recommendations, error = gnn_loader.get_user_recommendations(customer_id, limit)
        
        if error:
            print(f"❌ Recommendation failed: {error}")
            return jsonify({
                'status': 'error',
                'message': error,
                'customer_id': customer_id,
                'processing_time_seconds': round(time.time() - start_time, 2)
            }), 400
        
        # Get user history count
        user_history = gnn_loader.comments_df[
            gnn_loader.comments_df['customer_id'] == customer_id
        ]
        
        processing_time = round(time.time() - start_time, 2)
        print(f"✅ Recommendation completed in {processing_time}s")
        
        response_data = {
            'status': 'success',
            'customer_id': customer_id,
            'total_recommendations': len(recommendations),
            'requested_limit': limit,
            'user_history_count': len(user_history),
            'processing_time_seconds': processing_time,
            'algorithm': 'Graph Neural Network (GNN)',
            'model_info': {
                'r2_score': 0.8362,
                'total_users': len(gnn_loader.user_to_idx),
                'total_books': len(gnn_loader.book_to_idx)
            },
            'recommendations': recommendations
        }
        
        return jsonify(response_data), 200
        
    except Exception as e:
        processing_time = round(time.time() - start_time, 2)
        print(f"💥 Error after {processing_time}s: {str(e)}")
        import traceback
        traceback.print_exc()
        
        return jsonify({
            'status': 'error',
            'message': f'Internal server error: {str(e)}',
            'error_type': type(e).__name__,
            'processing_time_seconds': processing_time
        }), 500

if __name__ == '__main__':
    app.run(host='0.0.0.0', debug=True)