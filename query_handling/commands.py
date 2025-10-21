"""
System command handling for administrative operations.
"""

import io
import logging
from contextlib import redirect_stdout
from typing import Tuple

logger = logging.getLogger("RAGI")

class CommandHandler:
    """Handles system commands."""
    
    def __init__(self, rag_system):
        """Initialize with a reference to the RAG system."""
        self.rag_system = rag_system
        self.api_mode = False  # Flag to indicate if running via API
    
    def set_api_mode(self, api_mode: bool):
        """Set whether the handler is running in API mode."""
        self.api_mode = api_mode
    
    def handle_command(self, command: str) -> Tuple[str, bool]:
        """
        Handles system commands.
        
        Args:
            command: The command string
            
        Returns:
            Tuple of (response, should_continue)
        """
        command_lower = command.lower().strip()
        
        # For API mode, convert administrative commands to knowledge base queries
        api_blocked_commands = [
            'reindex', 'stats', 'help', 'clear', 'exit', 'quit', 'bye', 'goodbye',
            'new session', 'create session', 'start new chat', 'list sessions', 
            'show sessions', 'sessions', 'switch session', 'change session', 
            'load session', 'session stats', 'session statistics', 'clear session', 
            'reset session', 'semantic stats', 'semantic statistics', 'menu', 'commands'
        ]
        
        if self.api_mode and command_lower in api_blocked_commands:
            if command_lower in ['help', 'menu', 'commands']:
                return "I'm RAGI, your knowledge assistant! I can answer questions about policies, procedures, and services. What would you like to know?", True
            elif command_lower == 'stats':
                return "I have access to comprehensive information from the knowledge base including procedures, services, support, and more. What specific topic can I help you with?", True
            elif command_lower == 'reindex':
                return "I keep my knowledge base up to date automatically. Is there something specific from the knowledge base that I can help you find?", True
            elif command_lower in ['exit', 'quit', 'bye', 'goodbye']:
                return "Thanks for chatting! Feel free to ask me anything about the knowledge base anytime. Have a great day! 😊", True
            elif command_lower == 'clear':
                return "I'm ready for a fresh conversation! What would you like to know about the knowledge base?", True
            elif command_lower in ['new session', 'create session', 'start new chat']:
                return "Every conversation with me is fresh and focused! What topic can I help you with?", True
            elif command_lower in ['list sessions', 'show sessions', 'sessions']:
                return "I'm here to help you with any questions from the knowledge base. What would you like to know?", True
            elif command_lower in ['switch session', 'change session', 'load session']:
                return "You can always start a new topic with me! What information are you looking for?", True
            elif command_lower in ['session stats', 'session statistics']:
                return "I'm ready to provide you with information from the knowledge base. What specific area interests you?", True
            elif command_lower in ['clear session', 'reset session']:
                return "Starting fresh! How can I help you with information today?", True
            elif command_lower in ['semantic stats', 'semantic statistics']:
                return "I use advanced language processing to understand your questions from the knowledge base. What can I help you find?", True
            else:
                return "I'm here to help with questions from the knowledge base. What would you like to know?", True
        
        # Help command
        if command_lower in ["help", "menu", "commands"]:
            help_text = """
            Available Commands:
            - help: Display this help menu
            - exit, quit: Stop the application
            - clear: Reset the conversation memory
            - reindex: Reindex all documents
            - stats: See document statistics
            - semantic stats: Show semantic processor statistics
            - new session: Create a new chat session
            - list sessions: Show all available sessions
            - switch session: Change to a different session
            - session stats: Show session statistics
            - clear session: Clear current session memory
            - list folders: Show all available folders
            - filter folder <name>: Search only in specified folder
            - clear filter: Remove folder filter
            """
            return help_text, True
        
        # Exit commands
        elif command_lower in ["exit", "quit", "bye", "goodbye"]:
            return "Goodbye! Have a great day!", False
            
        # Clear memory command
        elif command_lower == "clear":
            # Reset memory but keep system message
            system_message = self.rag_system.memory.chat_memory.messages[0] if self.rag_system.memory.chat_memory.messages else None
            self.rag_system.memory.clear()
            if system_message:
                self.rag_system.memory.chat_memory.messages.append(system_message)
            return "Conversation memory has been reset.", True
            
        # Stats command
        elif command_lower == "stats":
            # Import here to avoid circular import
            from vector_management.manager import VectorStoreManager
            
            # Capture printed output
            f = io.StringIO()
            with redirect_stdout(f):
                VectorStoreManager.print_document_statistics(self.rag_system.vector_store)
            
            output = f.getvalue()
            if not output.strip():
                output = "No document statistics available."
                
            return output, True
            
        # Reindex command
        elif command_lower == "reindex":
            result = self.rag_system.reindex_documents()
            if result:
                return "Documents have been successfully reindexed.", True
            else:
                return "Failed to reindex documents. Check the log for details.", True

        # Session management commands
        elif command_lower in ["new session", "create session", "start new chat"]:
            success = self.rag_system.switch_session()
            if success:
                return "✅ Created new session successfully!", True
            else:
                return "❌ Failed to create new session.", True

        elif command_lower in ["list sessions", "show sessions", "sessions"]:
            self.rag_system.list_sessions_command()
            return "", True

        elif command_lower in ["switch session", "change session", "load session"]:
            success = self._handle_switch_session_interactive()
            if success:
                return "✅ Session switched successfully!", True
            else:
                return "❌ Failed to switch session.", True

        elif command_lower in ["session stats", "session statistics"]:
            self.rag_system.session_stats_command()
            return "", True

        elif command_lower in ["clear session", "reset session"]:
            self.rag_system.session_manager.clear_current_session_memory()
            return "✅ Session memory cleared!", True
        
        elif command_lower in ["semantic stats", "semantic statistics"]:
            return self._handle_semantic_stats(), True

        # Folder management commands
        elif command_lower in ["list folders", "show folders", "folders"]:
            return self._handle_list_folders(), True

        elif command_lower.startswith("filter folder "):
            folder_name = command[len("filter folder "):].strip()
            return self._handle_filter_folder(folder_name), True

        elif command_lower in ["clear filter", "remove filter", "reset filter"]:
            return self._handle_clear_filter(), True

        # Unknown command
        else:
            return f"Unknown command: {command}. Type 'help' to see available commands.", True

    def _handle_switch_session_interactive(self):
        """Handle interactive session switching."""
        try:
            self.rag_system.list_sessions_command()
            session_input = input("\nEnter session ID (first 8 chars) or press Enter to create new: ").strip()
            
            if not session_input:
                # Create new session
                return self.rag_system.switch_session()
            
            # Find full session ID from partial
            sessions = self.rag_system.session_manager.list_sessions()
            for session in sessions:
                if session.session_id.startswith(session_input):
                    return self.rag_system.switch_session(session.session_id)
            
            print(f"Session starting with '{session_input}' not found.")
            return False
            
        except Exception as e:
            logger.error(f"Error in interactive session switch: {e}")
            return False
    
    def _handle_semantic_stats(self) -> str:
        """Handle semantic processor statistics command."""
        try:
            # Get stats from input processor
            input_processor = getattr(self.rag_system, 'input_processor', None)
            if not input_processor:
                return "❌ Input processor not available"
            
            spacy_processor = getattr(input_processor, 'spacy_processor', None)
            if not spacy_processor:
                return "ℹ️ Semantic processor not enabled or not available"
            
            stats = spacy_processor.get_statistics()
            
            report = "\n📊 Semantic Processor Statistics:\n"
            report += "=" * 50 + "\n"
            report += f"🔧 Status: {'✅ Healthy' if spacy_processor.is_healthy() else '❌ Unhealthy'}\n"
            report += f"🔄 Initialized: {'✅ Yes' if stats['initialized'] else '❌ No'}\n"
            report += f"🧠 Model loaded: {'✅ Yes' if stats['model_loaded'] else '❌ No'}\n"
            report += f"📂 Domain clusters: {stats['domain_clusters']}\n"
            report += f"🔨 Grammar patterns: {stats['grammar_patterns']}\n"
            report += f"⚠️ Error count: {stats['error_count']}/{stats['max_errors']}\n"
            
            # Processing stats
            processing_stats = stats['stats']
            report += f"\n📈 Processing Statistics:\n"
            report += f"   • Queries processed: {processing_stats['queries_processed']}\n"
            report += f"   • Errors encountered: {processing_stats['errors_encountered']}\n"
            report += f"   • Avg processing time: {processing_stats['average_processing_time']:.3f}s\n"
            
            if processing_stats['last_error']:
                report += f"   • Last error: {processing_stats['last_error']}\n"
            
            return report
            
        except Exception as e:
            logger.error(f"Error getting semantic stats: {e}")
            return f"❌ Error retrieving semantic statistics: {e}"

    def _handle_list_folders(self) -> str:
        """Handle list folders command."""
        try:
            retrieval_handler = self.rag_system.retrieval_handler
            folder_info = retrieval_handler.get_available_folders()

            if not folder_info:
                return "❌ No folder information available. Make sure documents are indexed."

            report = "\n📁 Folder Structure:\n"
            report += "=" * 60 + "\n\n"

            # Show projects (top-level folders)
            if folder_info.get('projects'):
                report += "📂 Projects (Top-level folders):\n"
                for project, count in folder_info['projects'].items():
                    report += f"   • {project}: {count} documents\n"
                report += "\n"

            # Show all folders
            if folder_info.get('folders'):
                report += "📁 All Folders:\n"
                for folder, count in folder_info['folders'].items():
                    report += f"   • {folder}: {count} documents\n"
                report += "\n"

            # Show full paths (for nested folders)
            if folder_info.get('folder_paths'):
                report += "🗂️  Full Folder Paths:\n"
                for path, count in folder_info['folder_paths'].items():
                    report += f"   • {path}: {count} documents\n"

            report += f"\n📊 Total Documents: {folder_info.get('total_documents', 0)}\n"
            report += "\n💡 Use 'filter folder <name>' to search within a specific folder\n"

            return report

        except Exception as e:
            logger.error(f"Error listing folders: {e}")
            return f"❌ Error listing folders: {e}"

    def _handle_filter_folder(self, folder_name: str) -> str:
        """Handle filter folder command."""
        try:
            if not folder_name:
                return "❌ Please specify a folder name. Example: filter folder technical-docs"

            retrieval_handler = self.rag_system.retrieval_handler
            folder_info = retrieval_handler.get_available_folders()

            # Check if folder exists
            if folder_info.get('folders') and folder_name in folder_info['folders']:
                # Filter by folder_name
                retrieval_handler.set_folder_filter({'folder_name': folder_name})
                doc_count = folder_info['folders'][folder_name]
                return f"✅ Folder filter applied: '{folder_name}' ({doc_count} documents)\n" \
                       f"All queries will now search only in this folder.\n" \
                       f"Use 'clear filter' to remove the filter."

            elif folder_info.get('projects') and folder_name in folder_info['projects']:
                # Filter by project
                retrieval_handler.set_folder_filter({'project': folder_name})
                doc_count = folder_info['projects'][folder_name]
                return f"✅ Project filter applied: '{folder_name}' ({doc_count} documents)\n" \
                       f"All queries will now search only in this project.\n" \
                       f"Use 'clear filter' to remove the filter."

            elif folder_info.get('folder_paths') and folder_name in folder_info['folder_paths']:
                # Filter by full path
                retrieval_handler.set_folder_filter({'folder_path': folder_name})
                doc_count = folder_info['folder_paths'][folder_name]
                return f"✅ Path filter applied: '{folder_name}' ({doc_count} documents)\n" \
                       f"All queries will now search only in this path.\n" \
                       f"Use 'clear filter' to remove the filter."

            else:
                available = list(folder_info.get('folders', {}).keys())
                return f"❌ Folder '{folder_name}' not found.\n" \
                       f"Available folders: {', '.join(available[:5])}\n" \
                       f"Use 'list folders' to see all available folders."

        except Exception as e:
            logger.error(f"Error filtering folder: {e}")
            return f"❌ Error filtering folder: {e}"

    def _handle_clear_filter(self) -> str:
        """Handle clear filter command."""
        try:
            retrieval_handler = self.rag_system.retrieval_handler
            retrieval_handler.clear_folder_filter()
            return "✅ Folder filter cleared. Queries will now search all documents."

        except Exception as e:
            logger.error(f"Error clearing filter: {e}")
            return f"❌ Error clearing filter: {e}"