# app.py
from flask import Flask, render_template, jsonify, request
from game import ConquestGame, Player
import json

app = Flask(__name__)
game = ConquestGame()

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/api/game_state')
def get_game_state():
    # Convert graph to JSON format
    nodes = []
    for node in game.graph.nodes():
        nodes.append({
            'id': node,
            'owner': game.node_owners[node].value,
            'troops': game.node_troops[node]
        })
    
    edges = []
    for edge in game.graph.edges():
        edges.append({'source': edge[0], 'target': edge[1]})
    
    return jsonify({
        'nodes': nodes,
        'edges': edges,
        'scores': {k.value: v for k, v in game.get_scores().items()},
        'valid_moves': game.get_valid_moves(Player.BLUE),
        'game_over': game.game_over
    })

@app.route('/api/make_move', methods=['POST'])
def make_move():
    data = request.json
    success = game.make_move(
        Player.BLUE, 
        data['source'], 
        data['target'], 
        data['troops']
    )
    return jsonify({'success': success})

@app.route('/api/update')
def update_game():
    game.update()
    return jsonify({'status': 'updated'})

@app.route('/api/reset')
def reset_game():
    game.reset_game()
    return jsonify({'status': 'reset'})

if __name__ == '__main__':
    app.run(debug=True, port=5050)