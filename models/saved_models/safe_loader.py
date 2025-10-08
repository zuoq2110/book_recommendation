# Safe Model Loader for Advanced GNN Book Recommender
# Generated: 2025-09-09 23:42:21
# Timestamp: 234024

import torch
import pickle
import pandas as pd
import warnings
warnings.filterwarnings('ignore')

class SafeModelLoader:
    def __init__(self, timestamp="234024"):
        self.timestamp = timestamp
        self.model_data = None
        self.system_data = None
        
    def load_model(self):
        """Load model with safe settings"""
        print(f"📥 Loading model {self.timestamp}...")
        
        try:
            # Safe load for PyTorch 2.6+
            self.model_data = torch.load(
                f"saved_models/gnn_model_{self.timestamp}.pth", 
                map_location='cpu',
                weights_only=False  # Set to False for compatibility
            )
            print(f"✅ Model loaded - R²: {self.model_data['test_r2']:.4f}")
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False
    
    def load_data(self):
        """Load all supporting data"""
        print("📊 Loading supporting data...")
        
        try:
            # Load mappings
            with open(f"saved_models/user_to_idx_{self.timestamp}.pkl", 'rb') as f:
                user_to_idx = pickle.load(f)
            
            with open(f"saved_models/book_to_idx_{self.timestamp}.pkl", 'rb') as f:
                book_to_idx = pickle.load(f)
            
            # Load DataFrames
            books = pd.read_pickle(f"saved_models/books_{self.timestamp}.pkl")
            comments = pd.read_pickle(f"saved_models/comments_{self.timestamp}.pkl")
            
            # Load graph data (with safe globals if needed)
            try:
                gnn_data = torch.load(
                    f"saved_models/gnn_data_{self.timestamp}.pth", 
                    map_location='cpu',
                    weights_only=False
                )
            except:
                # Alternative safe loading
                import torch_geometric
                torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
                gnn_data = torch.load(
                    f"saved_models/gnn_data_{self.timestamp}.pth", 
                    map_location='cpu'
                )
            
            self.system_data = {
                'user_to_idx': user_to_idx,
                'book_to_idx': book_to_idx,
                'books': books,
                'comments': comments,
                'gnn_data': gnn_data
            }
            
            print(f"✅ Data loaded - {len(user_to_idx)} users, {len(books)} books")
            return True
            
        except Exception as e:
            print(f"❌ Error loading data: {e}")
            return False
    
    def get_model_info(self):
        """Display model information"""
        if not self.model_data:
            print("❌ Model not loaded yet!")
            return
        
        print("\n📊 MODEL INFORMATION:")
        print("=" * 30)
        print(f"Performance:")
        print(f"  • R² Score: {self.model_data['test_r2']:.4f}")
        print(f"  • RMSE: {self.model_data['test_rmse']:.4f}")
        print(f"  • MAE: {self.model_data['test_mae']:.4f}")
        print(f"  • Training Epochs: {self.model_data['epoch']}")
        
        config = self.model_data['model_config']
        print(f"\nArchitecture:")
        print(f"  • Embedding Dim: {config.get('embedding_dim', 'N/A')}")
        print(f"  • Hidden Dim: {config.get('hidden_dim', 'N/A')}")
        print(f"  • Layers: {config.get('num_layers', 'N/A')}")
        print(f"  • Dropout: {config.get('dropout', 'N/A')}")
        
        if self.system_data:
            print(f"\nData:")
            print(f"  • Users: {len(self.system_data['user_to_idx']):,}")
            print(f"  • Books: {len(self.system_data['books']):,}")
            print(f"  • Comments: {len(self.system_data['comments']):,}")
    
    def quick_recommendation_demo(self, customer_id=None):
        """Demo recommendation function"""
        if not self.system_data:
            print("❌ Data not loaded!")
            return
        
        # Pick a random customer if none provided
        if customer_id is None:
            import random
            customer_id = random.choice(list(self.system_data['user_to_idx'].keys()))
        
        print(f"\n🎯 DEMO RECOMMENDATION FOR CUSTOMER {customer_id}:")
        print("-" * 45)
        
        # Show customer's reading history
        customer_books = self.system_data['comments'][
            self.system_data['comments']['customer_id'] == customer_id
        ]
        
        if len(customer_books) == 0:
            print("❌ Customer has no reading history!")
            return
        
        print(f"📚 Customer đã đọc {len(customer_books)} cuốn sách")
        print(f"⭐ Rating trung bình: {customer_books['rating'].mean():.1f}/5")
        
        # Show top rated books by customer
        top_books = customer_books.nlargest(3, 'rating')
        print(f"\n🏆 Top 3 sách được rating cao nhất:")
        
        for idx, row in top_books.iterrows():
            product_id = row['product_id']
            rating = row['rating']
            
            book_info = self.system_data['books'][
                self.system_data['books']['product_id'] == product_id
            ]
            
            if not book_info.empty:
                book_title = book_info.iloc[0]['title']
                print(f"   ⭐ {rating}/5 - {book_title}")
        
        print("\n💡 Để tạo gợi ý mới, cần implement prediction logic")
        print("   (Requires model reconstruction with saved weights)")

# Example usage
if __name__ == "__main__":
    loader = SafeModelLoader()
    
    # Load model and data
    if loader.load_model() and loader.load_data():
        loader.get_model_info()
        loader.quick_recommendation_demo()
    else:
        print("❌ Failed to load model system")
