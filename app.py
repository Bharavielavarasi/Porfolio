import os
import logging
from flask import Flask, render_template, request, jsonify, send_from_directory
from pathlib import Path
import PyPDF2
import pdfplumber
import fitz  # PyMuPDF
import json
from datetime import datetime, date
from sentence_transformers import SentenceTransformer
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from typing import List, Dict, Tuple, Optional
from dotenv import load_dotenv
import re
from collections import defaultdict
import spacy
from dataclasses import dataclass
from difflib import SequenceMatcher
import string
import traceback

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

load_dotenv()

@dataclass
class KnowledgeChunk:
    content: str
    context: str
    category: str
    keywords: List[str]
    source: str
    confidence: float = 0.0
    raw_text: str = ""
    metadata: dict = None

class EnhancedPortfolioRAGChatbot:
    def __init__(self, static_folder: str, openai_api_key: str = None):
        self.static_folder = static_folder
        self.knowledge_chunks = []
        self.embeddings = []
        self.model = None
        self.nlp = None
        self.structured_data = {}
        self.raw_content = {}
        self.conversation_history = {}
        
        # Enhanced Portfolio data with COMPLETE information including dates
        self.portfolio_data = {
            'experience': {
                'ICSR IITM': {
                    'position': 'PowerBI and Python Developer',
                    'company': 'ICSR, IITM, Chennai',
                    'duration': '3 months',
                    'start_date': '2023-06-01',  # Added for precise date queries
                    'end_date': '2023-09-01',
                    'description': 'Developed and embedded interactive dashboards using Power BI Desktop and Power BI Service for data visualization. Built backend APIs with Python using the Flask framework and managed database operations through Microsoft SQL Server. Implemented role-based access control to ensure secure and customized access to reports. Designed a responsive web interface using HTML and CSS. Created a Python-based chatbot to assist users with frequently asked questions and provide guidance.',
                    'technologies': ['Power BI', 'Python', 'Flask', 'Microsoft SQL Server', 'HTML', 'CSS'],
                    'url': 'https://github.com/Bharavielavarasi/Intership'
                },
                'Afame Technologies': {
                    'position': 'Data Analyst',
                    'company': 'Afame Technologies',
                    'duration': '1 month',
                    'start_date': '2023-04-01',
                    'end_date': '2023-05-01',
                    'description': 'Conducted exploratory data analysis on HR and sales datasets to identify trends and patterns. Built interactive dashboards to support business decision-making processes. Developed automated data pipelines using Python to streamline data preparation and analysis workflows.',
                    'technologies': ['Python', 'Data Analysis', 'Dashboard Development', 'EDA'],
                    'url': 'https://colab.research.google.com/drive/10yjGLvDdx1-05ej2_uZ0dRzUpSEY6j_S'
                },
                'Cognify': {
                    'position': 'Data Scientist',
                    'company': 'Cognify',
                    'duration': '1 month',
                    'start_date': '2023-03-01',
                    'end_date': '2023-04-01',
                    'description': 'Analyzed a restaurant dataset to explore customer preferences, service features, and rating patterns. Performed data cleaning, statistical and geospatial analysis, and feature engineering. Built regression models to predict restaurant ratings and visualized insights on popular cuisines, top-rated cities, and the impact of table booking and online delivery on customer satisfaction.',
                    'technologies': ['Python', 'Machine Learning', 'Statistical Analysis', 'Data Visualization'],
                    'url': 'https://github.com/Bharavielavarasi/Intership'
                }
            },
            'projects': {
                'Traffic Optimization': {
                    'name': 'Machine Learning Project - Sustainable Traffic Optimization in Urban Areas',
                    'description': 'Built a CO₂ emissions prediction model using Multivariate Linear Regression, applied statistical methods (Fisher-Pearson Skewness, Jarque Bera) for validation. Integrated real-time traffic data and optimized routing for emissions reduction.',
                    'technologies': ['Python', 'Pandas', 'Scikit-learn', 'Statistical Analysis'],
                    'year': '2023',
                    'url': 'https://github.com/Bharavielavarasi/ML-project'
                },
                'Data Analysis': {
                    'name': 'Sales and HR Data Analysis',
                    'description': 'A comprehensive machine learning solution for business forecasting with interactive dashboards. Features multiple ML algorithms, real-time data processing, and beautiful visualizations. Advanced text processing application with sentiment analysis, entity recognition, and document classification.',
                    'technologies': ['Python', 'Scikit-learn', 'Plotly', 'Machine Learning'],
                    'year': '2023',
                    'url': 'https://github.com/Bharavielavarasi/Afame-Technologies-'
                },
                'Movie Analysis': {
                    'name': 'Movie Data Analysis',
                    'description': 'Analyzed movie industry data using SQL by joining multiple tables to extract insights on top-rated films, industry trends, actor demographics, release patterns, and financial performance. Applied grouping, filtering, aggregation, and currency conversion to generate actionable insights from complex datasets.',
                    'technologies': ['SQL', 'Data Analysis', 'Database Management'],
                    'year': '2023',
                    'url': 'https://github.com/Bharavielavarasi/Movie-data-analysis'
                },
                'Spotify Analysis': {
                    'name': 'Spotify Data Analysis Project',
                    'description': 'Analyzed Spotify data using Power BI to visualize music trends, artist performance, and listener preferences. Designed interactive dashboards showcasing top tracks, genre popularity, streaming patterns, and user engagement metrics.',
                    'technologies': ['Power BI', 'Data Visualization', 'Dashboard Design'],
                    'year': '2023',
                    'url': 'https://app.powerbi.com/groups/9211dc4d-d881-4c19-9518-84dc2aaccb2a/reports/91a7b345-0827-4fad-b7d6-95382cc60f53?ctid=45775dc0-2469-4389-9e79-b0919fcda527&pbi_source=linkShare'
                }
            },
            'education': {
                'degree': {
                    'name': 'Bachelor of Technology in Artificial Intelligence and Data Science',
                    'institution': 'Rajalakshmi Institute of Technology',
                    'status': 'Completed',
                    'field': 'Artificial Intelligence and Data Science',
                    'year': '2024',
                    'location': 'Chennai, Tamil Nadu'
                },
                '12th': {
                    'board': 'Central Board',
                    'school_name': 'KSR hi tech CBSE School',
                    'percentage': '85.3%',
                    'subjects': 'Science Stream',
                    'status': 'Completed',
                    'year': '2021',
                    'location': 'Tamil Nadu'
                },
                '10th': {
                    'board': 'Central Board',
                    'school_name': 'KSR hi tech CBSE School', 
                    'percentage': '83.2%',
                    'status': 'Completed',
                    'year': '2019',
                    'location': 'Tamil Nadu'
                }
            },
            'skills': {
                'programming': ['Python', 'SQL', 'HTML', 'CSS'],
                'data_science': ['Machine Learning', 'Deep Learning', 'Natural Language Processing', 'Computer Vision', 'Data Mining', 'Statistical Analysis', 'Predictive Modeling'],
                'frameworks': ['Flask','TensorFlow', 'PyTorch', 'Scikit-learn', 'Pandas', 'NumPy', 'Matplotlib', 'Seaborn', 'Plotly'],
                'databases': ['MySQL', 'Microsoft SQL Server', 'SQLite', 'PostgreSQL'],
                'tools': ['Power BI', 'Tableau', 'Git', 'GitHub', 'Jupyter', 'Google Colab', 'VS Code', 'PyCharm', 'Excel'],
                'cloud': ['AWS', 'Google Cloud Platform', 'Azure'],
                'specializations': ['Data Analysis', 'Business Intelligence', 'Data Visualization', 'ETL Processes', 'API Development', 'Web Development', 'Dashboard Development']
            },
            'contact': {
                'phone': '+91 9894663026',
                'email': 'bharavielavarasi@gmail.com',
                'linkedin': 'https://linkedin.com/in/bharavi-s',
                'github': 'https://github.com/Bharavielavarasi',
                'instagram': 'https://instagram.com/bharavi.03',
                'twitter': 'https://x.com/bharavi0310?t=NlpaNhS6Fy522zhnfazEoQ&s=09'
            },
            'basic_info': {
                'name': 'Bharavi',
                'full_name': 'Bharavi S',
                'designation': 'AI & Data Science Engineer',
                'bio': 'Passionate engineer leveraging machine learning, NLP, and data-driven insights to solve real-world problems, automate tasks, and enhance decision-making with intelligent systems.',
                'location': 'Tamil Nadu, India',
                'native': 'Perambalur, Tamil Nadu, India',
                'date_of_birth': '2003-10-03',  # Added for age calculations
                'age': self.calculate_age('2003-10-03')
            }
        }
        
        # Enhanced spell correction dictionary
        self.spell_corrections = {
            # Common typos
            'skils': 'skills', 'skilz': 'skills', 'skil': 'skill', 'skils':'skill',
            'experiance': 'experience', 'experence': 'experience', 'expirience': 'experience',
            'educaton': 'education', 'educaton': 'education', 'eductaion': 'education',
            'projets': 'projects', 'projet': 'project', 'projct': 'project',
            'contct': 'contact', 'cantact': 'contact', 'contat': 'contact',
            'loction': 'location', 'locaton': 'location', 'locaiton': 'location',
            'plce': 'place', 'palce': 'place', 'plac': 'place',
            'degre': 'degree', 'degri': 'degree', 'degrea': 'degree',
            'qualifcation': 'qualification', 'qualifiation': 'qualification',
            'univrsity': 'university', 'universty': 'university', 'univesity': 'university',
            'colege': 'college', 'collge': 'college', 'coleg': 'college',
            'scool': 'school', 'schol': 'school', 'shcool': 'school',
            'percentge': 'percentage', 'percentag': 'percentage', 'percntage': 'percentage',
            'compny': 'company', 'compani': 'company', 'copany': 'company',
            'positon': 'position', 'postion': 'position', 'positio': 'position',
            'intrship': 'internship', 'internshp': 'internship', 'intership': 'internship',
            'tecnology': 'technology', 'technolgy': 'technology', 'techology': 'technology',
            'programing': 'programming', 'programimg': 'programming', 'progaming': 'programming',
            'languag': 'language', 'langauge': 'language', 'languague': 'language',
            'framewrk': 'framework', 'framwork': 'framework', 'frameowrk': 'framework',
            'databas': 'database', 'databse': 'database', 'databae': 'database',
            'machin': 'machine', 'machne': 'machine', 'macine': 'machine',
            'learing': 'learning', 'leraning': 'learning', 'learnig': 'learning',
            'analys': 'analysis', 'anlaysis': 'analysis', 'analyis': 'analysis',
            'visualizaton': 'visualization', 'visualisaton': 'visualization',
            'emal': 'email', 'emil': 'email', 'emial': 'email',
            'phon': 'phone', 'fone': 'phone', 'phne': 'phone',
            'linkdin': 'linkedin', 'linkedn': 'linkedin', 'linkein': 'linkedin',
            'githb': 'github', 'gthub': 'github', 'gibhub': 'github',
            'nativ': 'native', 'natve': 'native', 'nativ': 'native',
            # Question words
            'wher': 'where', 'whre': 'where', 'wher': 'where',
            'wht': 'what', 'wat': 'what', 'whta': 'what',
            'hw': 'how', 'hwo': 'how', 'haw': 'how',
            'wich': 'which', 'whch': 'which', 'whih': 'which',
            # Age and date related
            'ag': 'age', 'agee': 'age', 'olds': 'old',
            'brth': 'birth', 'born': 'birth', 'dat': 'date'
        }
        
        self.initialize_models()
        self.load_and_process_portfolio_data()
    
    @staticmethod
    def calculate_age(birth_date_str: str) -> int:
        """Calculate current age from birth date string"""
        try:
            birth_date = datetime.strptime(birth_date_str, '%Y-%m-%d').date()
            today = date.today()
            age = today.year - birth_date.year - ((today.month, today.day) < (birth_date.month, birth_date.day))
            return age
        except:
            return 21  # fallback age

    def detect_query_intent(self, query: str) -> Dict[str, any]:
        """Detect specific intent and extract entities from user query"""
        query_lower = query.lower().strip()
        
        intent_data = {
            'intent': 'general',
            'entities': {},
            'specificity': 'general',
            'response_type': 'comprehensive'
        }
        
        # Personal information intents
        if any(term in query_lower for term in ['name', 'called', 'who are you', 'who is']):
            intent_data['intent'] = 'name'
            intent_data['response_type'] = 'specific'
            
        elif any(term in query_lower for term in ['age', 'old', 'years old']):
            intent_data['intent'] = 'age'
            intent_data['response_type'] = 'specific'
            
        elif any(term in query_lower for term in ['date of birth', 'birth date', 'born when', 'birthday']):
            intent_data['intent'] = 'birth_date'
            intent_data['response_type'] = 'specific'
            
        # Education intents with specificity
        elif any(term in query_lower for term in ['degree', 'bachelor', 'btech', 'graduation']):
            intent_data['intent'] = 'degree'
            intent_data['response_type'] = 'specific'
            if any(term in query_lower for term in ['college', 'university', 'institution', 'where']):
                intent_data['entities']['include_college'] = True
                
        elif any(term in query_lower for term in ['college', 'university', 'institution']):
            intent_data['intent'] = 'college'
            intent_data['response_type'] = 'specific'
            
        elif '12th' in query_lower or 'twelfth' in query_lower:
            intent_data['intent'] = '12th'
            intent_data['response_type'] = 'specific'
            if any(term in query_lower for term in ['school', 'where']):
                intent_data['entities']['include_school'] = True
            if any(term in query_lower for term in ['percentage', 'marks', 'score']):
                intent_data['entities']['include_percentage'] = True
                
        elif '10th' in query_lower or 'tenth' in query_lower:
            intent_data['intent'] = '10th'
            intent_data['response_type'] = 'specific'
            if any(term in query_lower for term in ['school', 'where']):
                intent_data['entities']['include_school'] = True
            if any(term in query_lower for term in ['percentage', 'marks', 'score']):
                intent_data['entities']['include_percentage'] = True
                
        elif any(term in query_lower for term in ['education', 'qualification', 'academic']):
            intent_data['intent'] = 'education'
            intent_data['response_type'] = 'comprehensive'
            
        # Experience intents with specificity
        elif any(term in query_lower for term in ['experience', 'work', 'job', 'internship']):
            intent_data['intent'] = 'experience'
            
            # Check for specific company names
            for exp_key, exp_data in self.portfolio_data['experience'].items():
                if exp_data['company'].lower() in query_lower or exp_key.lower() in query_lower:
                    intent_data['entities']['specific_company'] = exp_key
                    intent_data['response_type'] = 'specific'
                    break
            
            # Check for specific details requested
            if any(term in query_lower for term in ['description', 'details', 'what did', 'responsibilities']):
                intent_data['entities']['include_description'] = True
            if any(term in query_lower for term in ['duration', 'how long', 'period']):
                intent_data['entities']['include_duration'] = True
            if any(term in query_lower for term in ['technologies', 'tech', 'tools']):
                intent_data['entities']['include_technologies'] = True
                
        # Skills intents
        elif any(term in query_lower for term in ['skills', 'skill', 'technologies', 'technical']):
            intent_data['intent'] = 'skills'
            if any(term in query_lower for term in ['all', 'complete', 'list']):
                intent_data['response_type'] = 'comprehensive'
            else:
                intent_data['response_type'] = 'specific'
                
        # Project intents
        elif any(term in query_lower for term in ['project', 'projects', 'built', 'developed']):
            intent_data['intent'] = 'projects'
            intent_data['response_type'] = 'comprehensive'
            
        # Contact intents
        elif any(term in query_lower for term in ['contact', 'reach', 'email', 'phone', 'linkedin']):
            intent_data['intent'] = 'contact'
            
            # Specific contact method
            if 'email' in query_lower:
                intent_data['entities']['contact_type'] = 'email'
                intent_data['response_type'] = 'specific'
            elif 'phone' in query_lower:
                intent_data['entities']['contact_type'] = 'phone'
                intent_data['response_type'] = 'specific'
            elif 'linkedin' in query_lower:
                intent_data['entities']['contact_type'] = 'linkedin'
                intent_data['response_type'] = 'specific'
            else:
                intent_data['response_type'] = 'comprehensive'
                
        # Location intents
        elif any(term in query_lower for term in ['location', 'place', 'where', 'from', 'native']):
            intent_data['intent'] = 'location'
            if any(term in query_lower for term in ['native', 'from', 'hometown']):
                intent_data['entities']['location_type'] = 'native'
            else:
                intent_data['entities']['location_type'] = 'current'
            intent_data['response_type'] = 'specific'
        
        return intent_data

    def generate_precise_response(self, query: str, intent_data: Dict, relevant_chunks: List[Tuple[KnowledgeChunk, float]]) -> str:
        """Generate precise, context-aware responses based on intent"""
        
        intent = intent_data['intent']
        entities = intent_data.get('entities', {})
        response_type = intent_data['response_type']
        
        # Handle specific intents with precise responses
        if intent == 'name':
            return f"Her name is {self.portfolio_data['basic_info']['name']}."
            
        elif intent == 'age':
            age = self.portfolio_data['basic_info']['age']
            return f"She is {age} years old."
            
        elif intent == 'birth_date':
            birth_date = self.portfolio_data['basic_info']['date_of_birth']
            formatted_date = datetime.strptime(birth_date, '%Y-%m-%d').strftime('%B %d, %Y')
            return f"She was born on {formatted_date}."
            
        elif intent == 'degree':
            degree_info = self.portfolio_data['education']['degree']
            response = f"She completed {degree_info['name']}"
            if entities.get('include_college'):
                response += f" from {degree_info['institution']}"
            response += "."
            return response
            
        elif intent == 'college':
            college_info = self.portfolio_data['education']['degree']
            return f"She studied at {college_info['institution']}, {college_info['location']}."
            
        elif intent == '12th':
            grade_12 = self.portfolio_data['education']['12th']
            response_parts = []
            
            if entities.get('include_percentage'):
                response_parts.append(f"She scored {grade_12['percentage']} in 12th standard")
            else:
                response_parts.append(f"She completed 12th standard with {grade_12['percentage']}")
                
            if entities.get('include_school'):
                response_parts.append(f"from {grade_12['school_name']}")
            else:
                response_parts.append(f"from {grade_12['school_name']} ({grade_12['board']})")
                
            return " ".join(response_parts) + "."
            
        elif intent == '10th':
            grade_10 = self.portfolio_data['education']['10th']
            response_parts = []
            
            if entities.get('include_percentage'):
                response_parts.append(f"She scored {grade_10['percentage']} in 10th standard")
            else:
                response_parts.append(f"She completed 10th standard with {grade_10['percentage']}")
                
            if entities.get('include_school'):
                response_parts.append(f"from {grade_10['school_name']}")
            else:
                response_parts.append(f"from {grade_10['school_name']} ({grade_10['board']})")
                
            return " ".join(response_parts) + "."
            
        elif intent == 'experience':
            if entities.get('specific_company'):
                # Handle specific company queries
                company_key = entities['specific_company']
                exp_data = self.portfolio_data['experience'][company_key]
                
                response_parts = [f"At {exp_data['company']}, she worked as a {exp_data['position']}"]
                
                if entities.get('include_duration'):
                    response_parts.append(f"for {exp_data['duration']}")
                    
                if entities.get('include_description'):
                    clean_desc = self.remove_personal_pronouns_comprehensive(exp_data['description'])
                    response_parts.append(f". {clean_desc}")
                    
                if entities.get('include_technologies'):
                    response_parts.append(f" Technologies used: {', '.join(exp_data['technologies'])}")
                    
                return " ".join(response_parts) + "."
            else:
                # General experience overview
                exp_data = self.portfolio_data['experience']
                companies = [data['company'] for data in exp_data.values()]
                positions = [data['position'] for data in exp_data.values()]
                
                return f"She has worked at {len(companies)} companies: {', '.join(companies)} in roles including {', '.join(set(positions))}."
                
        elif intent == 'contact':
            contact_info = self.portfolio_data['contact']
            
            if entities.get('contact_type'):
                contact_type = entities['contact_type']
                if contact_type in contact_info:
                    return f"Her {contact_type}: {contact_info[contact_type]}"
            else:
                # Comprehensive contact info
                response_parts = ["You can reach her through:"]
                response_parts.append(f"Email: {contact_info['email']}")
                response_parts.append(f"Phone: {contact_info['phone']}")
                response_parts.append(f"LinkedIn: {contact_info['linkedin']}")
                return " ".join(response_parts) + "."
                
        elif intent == 'location':
            basic_info = self.portfolio_data['basic_info']
            location_type = entities.get('location_type', 'current')
            
            if location_type == 'native':
                return f"She is originally from {basic_info['native']}."
            else:
                return f"She currently lives in {basic_info['location']}."
                
        elif intent == 'skills':
            if response_type == 'comprehensive':
                return self.format_conversational_skills_response()
            else:
                # Return a summary
                all_skills = []
                for category, skills in self.portfolio_data['skills'].items():
                    all_skills.extend(skills)
                unique_skills = sorted(set(all_skills))
                return f"Her key skills include: {', '.join(unique_skills[:10])}."
                
        elif intent == 'education':
            return self.format_conversational_education_response(relevant_chunks)
            
        elif intent == 'projects':
            return self.format_conversational_project_response(relevant_chunks)
        
        # Fallback to original response generation
        return self.generate_comprehensive_response(query, relevant_chunks)

    def advanced_spell_correction(self, text: str) -> Tuple[str, bool, List[str]]:
        """Advanced spell correction with multiple techniques"""
        corrected = text.lower()
        was_corrected = False
        corrections_made = []
        
        # Split into words while preserving punctuation
        words = re.findall(r'\b\w+\b|\W+', corrected)
        corrected_words = []
        
        for word in words:
            if word in self.spell_corrections:
                corrected_words.append(self.spell_corrections[word])
                corrections_made.append(f"{word} → {self.spell_corrections[word]}")
                was_corrected = True
            elif word.isalpha() and len(word) > 3:
                # Check for common patterns
                best_match = self.find_closest_word(word)
                if best_match and best_match != word:
                    corrected_words.append(best_match)
                    corrections_made.append(f"{word} → {best_match}")
                    was_corrected = True
                else:
                    corrected_words.append(word)
            else:
                corrected_words.append(word)
        
        return ''.join(corrected_words), was_corrected, corrections_made
    
    def find_closest_word(self, word: str) -> Optional[str]:
        """Find closest word using similarity matching"""
        common_words = list(self.spell_corrections.values()) + [
            'skills', 'experience', 'education', 'projects', 'contact', 'location',
            'degree', 'qualification', 'university', 'college', 'school', 'percentage',
            'company', 'position', 'internship', 'technology', 'programming', 'language',
            'framework', 'database', 'machine', 'learning', 'analysis', 'visualization',
            'age', 'birth', 'date', 'name'
        ]
        
        best_match = None
        best_ratio = 0.7  # Minimum similarity threshold
        
        for candidate in common_words:
            ratio = SequenceMatcher(None, word, candidate).ratio()
            if ratio > best_ratio:
                best_match = candidate
                best_ratio = ratio
        
        return best_match

    def remove_personal_pronouns_comprehensive(self, response: str) -> str:
        """Comprehensive personal pronoun removal with better context handling"""
        if not response:
            return response
        
        # Step 1: Convert possessive structures
        response = re.sub(r'\bmy\s+([a-zA-Z]+)', r'the \1', response, flags=re.IGNORECASE)
        response = re.sub(r'\bhis\s+([a-zA-Z]+)', r'the \1', response, flags=re.IGNORECASE)
        response = re.sub(r'\bher\s+([a-zA-Z]+)', r'the \1', response, flags=re.IGNORECASE)
        
        # Step 2: Remove subject pronouns with context
        response = re.sub(r'\bi\s+am\b', 'She is', response, flags=re.IGNORECASE)
        response = re.sub(r'\bi\s+have\b', 'She has', response, flags=re.IGNORECASE)
        response = re.sub(r'\bi\s+work\b', 'She works', response, flags=re.IGNORECASE)
        response = re.sub(r'\bi\s+study\b', 'She studied', response, flags=re.IGNORECASE)
        response = re.sub(r'\bi\s+develop\b', 'She developed', response, flags=re.IGNORECASE)
        response = re.sub(r'\bi\s+build\b', 'She built', response, flags=re.IGNORECASE)
        response = re.sub(r'\bi\s+create\b', 'She created', response, flags=re.IGNORECASE)
        response = re.sub(r'\bi\s+analyze\b', 'She analyzed', response, flags=re.IGNORECASE)
        response = re.sub(r'\bi\s+design\b', 'She designed', response, flags=re.IGNORECASE)
        response = re.sub(r'\bi\s+implement\b', 'She implemented', response, flags=re.IGNORECASE)
        
        # Step 3: Remove remaining pronouns
        response = re.sub(r'\bi\b', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\bme\b', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\bmyself\b', '', response, flags=re.IGNORECASE)
        response = re.sub(r'\bmine\b', '', response, flags=re.IGNORECASE)
        
        # Step 4: Clean up formatting
        response = re.sub(r'\s+', ' ', response)  # Multiple spaces
        response = re.sub(r'\s*,\s*', ', ', response)  # Comma spacing
        response = re.sub(r'\s*\.\s*', '. ', response)  # Period spacing
        response = re.sub(r'^\s*[,.]', '', response)  # Leading punctuation
        response = re.sub(r'\.\s*,', '.', response)  # Trailing period
        
        # Step 5: Fix capitalization
        sentences = response.split('. ')
        capitalized_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence:
                sentence = sentence[0].upper() + sentence[1:] if len(sentence) > 1 else sentence.upper()
                capitalized_sentences.append(sentence)
        
        return '. '.join(capitalized_sentences).strip()

    def remove_duplicate_content(self, text: str) -> str:
        """Remove duplicate sentences and phrases from text"""
        sentences = text.split('. ')
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            sentence = sentence.strip()
            if sentence and sentence.lower() not in seen:
                seen.add(sentence.lower())
                unique_sentences.append(sentence)
        
        return '. '.join(unique_sentences)

    def initialize_models(self):
        """Initialize ML models and NLP components"""
        try:
            self.model = SentenceTransformer('all-MiniLM-L6-v2')
            logging.info("✅ Sentence transformer model loaded successfully")
            
            try:
                self.nlp = spacy.load("en_core_web_sm")
                logging.info("✅ spaCy model loaded successfully")
            except OSError:
                logging.warning("⚠️ spaCy model not found. Install with: python -m spacy download en_core_web_sm")
                self.nlp = None
                
        except Exception as e:
            logging.error(f"❌ Failed to load models: {e}")

    def extract_keywords_dynamically(self, text: str) -> List[str]:
        """Enhanced keyword extraction"""
        keywords = []
        text_lower = text.lower()
        
        # Specific domain keywords
        domain_keywords = {
            'education': ['degree', 'bachelor', 'btech', 'engineering', 'university', 'college', 'school', '10th', '12th', 'percentage', 'cgpa', 'marks'],
            'skills': ['python', 'sql', 'machine learning', 'data science', 'flask', 'html', 'css', 'power bi', 'tableau'],
            'experience': ['developer', 'analyst', 'scientist', 'internship', 'company', 'position', 'work'],
            'projects': ['project', 'analysis', 'model', 'dashboard', 'application', 'system'],
            'contact': ['email', 'phone', 'linkedin', 'github', 'contact'],
            'personal': ['name', 'location', 'native', 'date', 'birth', 'place', 'age', 'old']
        }
        
        for category, category_keywords in domain_keywords.items():
            if any(kw in text_lower for kw in category_keywords):
                keywords.extend([kw for kw in category_keywords if kw in text_lower])
        
        if self.nlp:
            doc = self.nlp(text_lower)
            for token in doc:
                if (token.pos_ in ['NOUN', 'PROPN', 'NUM'] and 
                    not token.is_stop and 
                    not token.is_punct and 
                    len(token.text) > 2):
                    keywords.append(token.lemma_)
            
            for ent in doc.ents:
                if ent.label_ in ['PERSON', 'ORG', 'PRODUCT', 'GPE', 'LANGUAGE', 'DATE', 'PERCENT']:
                    keywords.append(ent.text.lower())
        
        return list(set(keywords))

    def _categorize_content(self, content: str) -> str:
        """Enhanced content categorization"""
        content_lower = content.lower()
        
        category_patterns = {
            'personal': ['name', 'bharavi', 'date of birth', 'born', 'personal', 'about', 'bio', 'age', 'old'],
            'location': ['location', 'native', 'place', 'tamil nadu', 'india', 'chennai', 'perambalur', 'from'],
            'education': ['education', 'degree', 'bachelor', 'btech', 'university', 'college', 'school', '10th', '12th', 'percentage', 'cgpa', 'marks'],
            'skills': ['skills', 'programming', 'python', 'sql', 'machine learning', 'data science', 'technology', 'tools', 'frameworks'],
            'experience': ['experience', 'work', 'internship', 'position', 'company', 'developer', 'analyst', 'scientist'],
            'projects': ['project', 'analysis', 'model', 'dashboard', 'application', 'built', 'developed', 'created'],
            'contact': ['contact', 'email', 'phone', 'linkedin', 'github', 'social']
        }
        
        scores = {}
        for category, patterns in category_patterns.items():
            score = sum(1 for pattern in patterns if pattern in content_lower)
            if score > 0:
                scores[category] = score
        
        return max(scores.items(), key=lambda x: x[1])[0] if scores else 'general'

    def load_and_process_portfolio_data(self):
        """Load and process all portfolio data with GUARANTEED chunks"""
        logging.info("🔄 Starting portfolio data loading...")
        
        # First priority: Create portfolio data chunks (MUST have these)
        portfolio_chunks = self.create_comprehensive_portfolio_chunks()
        all_chunks = portfolio_chunks
        logging.info(f"✅ Created {len(portfolio_chunks)} chunks from portfolio data")
        
        # Second priority: Try to load PDF files (optional)
        pdf_files = [
            os.path.join(self.static_folder, 'about.pdf'),
            os.path.join(self.static_folder, 'resume.pdf')
        ]
        
        pdf_chunks_count = 0
        for pdf_file in pdf_files:
            if os.path.exists(pdf_file):
                logging.info(f"📄 Processing {os.path.basename(pdf_file)}")
                content = self.extract_text_from_pdf(pdf_file)
                if content:
                    chunks = self.create_enhanced_chunks(content, os.path.basename(pdf_file))
                    all_chunks.extend(chunks)
                    pdf_chunks_count += len(chunks)
                    logging.info(f"✅ Created {len(chunks)} chunks from {os.path.basename(pdf_file)}")
            else:
                logging.warning(f"⚠️ PDF file not found: {os.path.basename(pdf_file)}")
        
        self.knowledge_chunks = all_chunks
        
        # Create embeddings if model is available
        if self.model and self.knowledge_chunks:
            try:
                chunk_contents = [chunk.content for chunk in self.knowledge_chunks]
                self.embeddings = self.model.encode(chunk_contents)
                logging.info(f"✅ Created embeddings for {len(self.knowledge_chunks)} chunks")
            except Exception as e:
                logging.error(f"❌ Failed to create embeddings: {e}")
        
        # Final verification
        total_chunks = len(self.knowledge_chunks)
        logging.info(f"📊 FINAL CHUNK SUMMARY:")
        logging.info(f"   📁 Portfolio chunks: {len(portfolio_chunks)}")
        logging.info(f"   📄 PDF chunks: {pdf_chunks_count}")
        logging.info(f"   📚 Total chunks: {total_chunks}")
        
        if total_chunks == 0:
            logging.error("❌ CRITICAL: No knowledge chunks created!")
        else:
            logging.info("✅ Knowledge base loaded successfully!")

    def create_comprehensive_portfolio_chunks(self) -> List[KnowledgeChunk]:
        """Create comprehensive chunks from portfolio data - GUARANTEED to work with enhanced personal info"""
        chunks = []
        
        try:
            # 1. PERSONAL INFORMATION CHUNKS (Enhanced with age and birth date)
            basic_info = self.portfolio_data['basic_info']
            
            # Name chunk
            name_content = f"Her name is {basic_info['name']}"
            chunks.append(KnowledgeChunk(
                content=name_content,
                context="personal information",
                category="personal",
                keywords=["name", "bharavi"],
                source="portfolio",
                raw_text=name_content,
                metadata={'type': 'basic_info', 'field': 'name'}
            ))
            
            # Age chunk
            age_content = f"She is {basic_info['age']} years old"
            chunks.append(KnowledgeChunk(
                content=age_content,
                context="age information",
                category="personal",
                keywords=["age", "years", "old"],
                source="portfolio",
                raw_text=age_content,
                metadata={'type': 'basic_info', 'field': 'age'}
            ))
            
            # Date of birth chunk
            birth_date = datetime.strptime(basic_info['date_of_birth'], '%Y-%m-%d').strftime('%B %d, %Y')
            birth_content = f"She was born on {birth_date}"
            chunks.append(KnowledgeChunk(
                content=birth_content,
                context="birth date information",
                category="personal",
                keywords=["birth", "date", "born", "birthday"],
                source="portfolio",
                raw_text=birth_content,
                metadata={'type': 'basic_info', 'field': 'birth_date'}
            ))
            
            # Designation chunk
            designation_content = f"Bharavi is an {basic_info['designation']}"
            chunks.append(KnowledgeChunk(
                content=designation_content,
                context="professional information",
                category="personal",
                keywords=["designation", "role", "position", "ai", "data science", "engineer"],
                source="portfolio",
                raw_text=designation_content,
                metadata={'type': 'basic_info', 'field': 'designation'}
            ))
            
            # Bio chunk
            bio_content = self.remove_personal_pronouns_comprehensive(basic_info['bio'])
            chunks.append(KnowledgeChunk(
                content=bio_content,
                context="professional summary",
                category="personal",
                keywords=["engineer", "machine learning", "nlp", "data science", "passionate"],
                source="portfolio",
                raw_text=bio_content,
                metadata={'type': 'basic_info', 'field': 'bio'}
            ))
            
            # 2. EDUCATION CHUNKS (Enhanced with school names)
            education_data = self.portfolio_data['education']
            
            # Degree information chunk
            if 'degree' in education_data:
                degree_info = education_data['degree']
                degree_content = f"She completed {degree_info['name']} at {degree_info['institution']}."
                chunks.append(KnowledgeChunk(
                    content=degree_content,
                    context="degree information",
                    category="education",
                    keywords=["degree", "bachelor", "btech", "engineering", "college", "university", "artificial intelligence", "data science"],
                    source="portfolio",
                    raw_text=degree_content,
                    metadata={'type': 'education', 'level': 'degree'}
                ))
                
                # Separate college chunk for "college" queries
                college_content = f"She studied at {degree_info['institution']}, {degree_info['location']}. Completed {degree_info['name']}."
                chunks.append(KnowledgeChunk(
                    content=college_content,
                    context="college information",
                    category="education", 
                    keywords=["college", "university", "institution", "studies", "engineering", "rajalakshmi"],
                    source="portfolio",
                    raw_text=college_content,
                    metadata={'type': 'education', 'level': 'college'}
                ))
            
            # 12th grade chunk with school name
            if '12th' in education_data:
                grade_12_info = education_data['12th']
                grade_12_content = f"She completed 12th standard with {grade_12_info['percentage']} in {grade_12_info['subjects']} from {grade_12_info['school_name']} ({grade_12_info['board']})."
                chunks.append(KnowledgeChunk(
                    content=grade_12_content,
                    context="12th grade information",
                    category="education",
                    keywords=["12th", "twelfth", "percentage", "science", "board", "school"],
                    source="portfolio",
                    raw_text=grade_12_content,
                    metadata={'type': 'education', 'level': '12th'}
                ))
            
            # 10th grade chunk with school name
            if '10th' in education_data:
                grade_10_info = education_data['10th']
                grade_10_content = f"She completed 10th standard with {grade_10_info['percentage']} from {grade_10_info['school_name']} ({grade_10_info['board']})."
                chunks.append(KnowledgeChunk(
                    content=grade_10_content,
                    context="10th grade information",
                    category="education",
                    keywords=["10th", "tenth", "percentage", "board", "school"],
                    source="portfolio",
                    raw_text=grade_10_content,
                    metadata={'type': 'education', 'level': '10th'}
                ))
            
            # 3. LOCATION CHUNKS
            current_location_content = f"Bharavi currently lives in {basic_info['location']}"
            chunks.append(KnowledgeChunk(
                content=current_location_content,
                context="location information",
                category="location",
                keywords=["location", "current", "live", "tamil nadu", "india"],
                source="portfolio",
                raw_text=current_location_content,
                metadata={'type': 'location', 'field': 'current'}
            ))
            
            native_place_content = f"Bharavi is originally from {basic_info['native']}"
            chunks.append(KnowledgeChunk(
                content=native_place_content,
                context="native place information",
                category="location",
                keywords=["native", "place", "from", "perambalur", "tamil nadu", "origin"],
                source="portfolio",
                raw_text=native_place_content,
                metadata={'type': 'location', 'field': 'native'}
            ))
            
            # 4. COMPLETE SKILLS CHUNKS
            all_skills_list = []
            for category, skills_list in self.portfolio_data['skills'].items():
                all_skills_list.extend(skills_list)
                
                # Individual category chunks
                category_skills_content = f"{category.replace('_', ' ').title()} skills: {', '.join(skills_list)}"
                chunks.append(KnowledgeChunk(
                    content=category_skills_content,
                    context=f"{category} skills category",
                    category="skills",
                    keywords=["skills", category.replace('_', ' ')] + [skill.lower() for skill in skills_list],
                    source="portfolio",
                    raw_text=category_skills_content,
                    metadata={'type': 'skills', 'category': category}
                ))
            
            # All skills comprehensive chunk
            unique_skills = sorted(set(all_skills_list))
            all_skills_content = f"Bharavi's complete skill set includes: {', '.join(unique_skills)}"
            chunks.append(KnowledgeChunk(
                content=all_skills_content,
                context="complete skills overview",
                category="skills",
                keywords=["all skills", "complete skills", "technical skills"] + [skill.lower() for skill in unique_skills],
                source="portfolio",
                raw_text=all_skills_content,
                metadata={'type': 'skills', 'category': 'all'}
            ))
            
            # 5. ENHANCED EXPERIENCE CHUNKS with dates and detailed info
            for exp_key, exp_data in self.portfolio_data['experience'].items():
                # Company and position chunk
                position_content = f"Bharavi worked as {exp_data['position']} at {exp_data['company']} for {exp_data.get('duration', 'some time')}"
                chunks.append(KnowledgeChunk(
                    content=position_content,
                    context="work experience position",
                    category="experience",
                    keywords=["position", "company", "experience", exp_data['position'].lower(), exp_data['company'].lower()],
                    source="portfolio",
                    raw_text=position_content,
                    metadata={'type': 'experience', 'company': exp_key, 'field': 'position'}
                ))
                
                # Detailed description chunk
                clean_description = self.remove_personal_pronouns_comprehensive(exp_data['description'])
                description_content = f"At {exp_data['company']} as {exp_data['position']}: {clean_description}"
                chunks.append(KnowledgeChunk(
                    content=description_content,
                    context="work experience details",  
                    category="experience",
                    keywords=self.extract_keywords_dynamically(exp_data['description']) + [exp_data['company'].lower()],
                    source="portfolio",
                    raw_text=description_content,
                    metadata={'type': 'experience', 'company': exp_key, 'field': 'description'}
                ))
                
                # Duration specific chunk
                if 'duration' in exp_data:
                    duration_content = f"She worked at {exp_data['company']} for {exp_data['duration']}"
                    chunks.append(KnowledgeChunk(
                        content=duration_content,
                        context="experience duration",
                        category="experience",
                        keywords=["duration", "period", "time", exp_data['company'].lower()],
                        source="portfolio",
                        raw_text=duration_content,
                        metadata={'type': 'experience', 'company': exp_key, 'field': 'duration'}
                    ))
                
                # Technologies used chunk
                if 'technologies' in exp_data:
                    tech_content = f"Technologies used at {exp_data['company']}: {', '.join(exp_data['technologies'])}"
                    chunks.append(KnowledgeChunk(
                        content=tech_content,
                        context="experience technologies",
                        category="experience",
                        keywords=["technologies"] + [tech.lower() for tech in exp_data['technologies']],
                        source="portfolio",
                        raw_text=tech_content,
                        metadata={'type': 'experience', 'company': exp_key, 'field': 'technologies'}
                    ))
            
            # 6. PROJECT CHUNKS (Enhanced)
            for proj_key, proj_data in self.portfolio_data['projects'].items():
                # Project name chunk
                project_name_content = f"Project: {proj_data['name']}"
                chunks.append(KnowledgeChunk(
                    content=project_name_content,
                    context="project information",
                    category="projects",
                    keywords=["project", "name"] + self.extract_keywords_dynamically(proj_data['name']),
                    source="portfolio",
                    raw_text=project_name_content,
                    metadata={'type': 'projects', 'project': proj_key, 'field': 'name'}
                ))
                
                # Project description chunk
                clean_project_desc = self.remove_personal_pronouns_comprehensive(proj_data['description'])
                project_desc_content = f"{proj_data['name']}: {clean_project_desc}"
                chunks.append(KnowledgeChunk(
                    content=project_desc_content,
                    context="project details",
                    category="projects",
                    keywords=self.extract_keywords_dynamically(proj_data['description']),
                    source="portfolio",
                    raw_text=project_desc_content,
                    metadata={'type': 'projects', 'project': proj_key, 'field': 'description'}
                ))
                
                # Project technologies chunk
                if 'technologies' in proj_data:
                    project_tech_content = f"Technologies used in {proj_data['name']}: {', '.join(proj_data['technologies'])}"
                    chunks.append(KnowledgeChunk(
                        content=project_tech_content,
                        context="project technologies",
                        category="projects",
                        keywords=["technologies"] + [tech.lower() for tech in proj_data['technologies']],
                        source="portfolio",
                        raw_text=project_tech_content,
                        metadata={'type': 'projects', 'project': proj_key, 'field': 'technologies'}
                    ))
            
            # 7. CONTACT INFORMATION CHUNKS (Enhanced)
            contact_info = self.portfolio_data['contact']
            for contact_type, contact_value in contact_info.items():
                contact_content = f"Her {contact_type}: {contact_value}"
                chunks.append(KnowledgeChunk(
                    content=contact_content,
                    context="contact information",
                    category="contact",
                    keywords=[contact_type, "contact", "reach"],
                    source="portfolio",
                    raw_text=contact_content,
                    metadata={'type': 'contact', 'method': contact_type}
                ))
            
            logging.info(f"✅ Successfully created {len(chunks)} comprehensive portfolio chunks with enhanced personal info")
            return chunks
            
        except Exception as e:
            logging.error(f"❌ Error creating portfolio chunks: {e}")
            # Return at least basic chunks if something fails
            fallback_chunks = [
                KnowledgeChunk(
                    content="Bharavi is pursuing Bachelor of Technology in Artificial Intelligence and Data Science",
                    context="education",
                    category="education", 
                    keywords=["degree", "bachelor", "btech", "artificial intelligence", "data science"],
                    source="fallback",
                    raw_text="Bharavi is pursuing Bachelor of Technology in Artificial Intelligence and Data Science"
                ),
                KnowledgeChunk(
                    content="Bharavi has skills in Python, SQL, Machine Learning, Data Science, Flask, Power BI and more",
                    context="skills",
                    category="skills",
                    keywords=["skills", "python", "sql", "machine learning", "data science"],
                    source="fallback", 
                    raw_text="Bharavi has skills in Python, SQL, Machine Learning, Data Science, Flask, Power BI and more"
                )
            ]
            logging.info(f"⚠️ Using {len(fallback_chunks)} fallback chunks")
            return fallback_chunks

    def extract_text_from_pdf(self, pdf_path: str) -> str:
        """Enhanced PDF text extraction"""
        if not os.path.exists(pdf_path):
            logging.error(f"❌ PDF file not found: {pdf_path}")
            return ""
        
        try:
            extracted_text = self._extract_with_pdfplumber_detailed(pdf_path)
            
            if not extracted_text.strip():
                extracted_text = self._extract_with_pymupdf(pdf_path)
            
            if not extracted_text.strip():
                extracted_text = self._extract_with_pypdf2(pdf_path)
            
            if extracted_text.strip():
                self.raw_content[pdf_path] = extracted_text
                logging.info(f"✅ Successfully extracted {len(extracted_text)} characters from {pdf_path}")
                return extracted_text
            else:
                logging.error(f"❌ All extraction methods failed for {pdf_path}")
                return ""
                
        except Exception as e:
            logging.error(f"❌ Critical error during PDF extraction: {e}")
            return ""
    
    def _extract_with_pdfplumber_detailed(self, pdf_path: str) -> str:
        """Extract text using pdfplumber"""
        try:
            import pdfplumber
            text_lines = []
            with pdfplumber.open(pdf_path) as pdf:
                for page_num, page in enumerate(pdf.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            lines = page_text.split('\n')
                            for line in lines:
                                line = line.strip()
                                if line:
                                    text_lines.append(line)
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to extract page {page_num + 1}: {e}")
                        continue
            return '\n'.join(text_lines)
        except ImportError:
            logging.warning("⚠️ pdfplumber not installed")
            return ""
        except Exception as e:
            logging.error(f"❌ pdfplumber extraction error: {e}")
            return ""
    
    def _extract_with_pymupdf(self, pdf_path: str) -> str:
        """Extract text using PyMuPDF"""
        try:
            import fitz
            text_lines = []
            pdf_document = fitz.open(pdf_path)
            
            for page_num in range(pdf_document.page_count):
                try:
                    page = pdf_document[page_num]
                    page_text = page.get_text()
                    if page_text:
                        lines = page_text.split('\n')
                        for line in lines:
                            line = line.strip()
                            if line:
                                text_lines.append(line)
                except Exception as e:
                    logging.warning(f"⚠️ Failed to extract page {page_num + 1}: {e}")
                    continue
            
            pdf_document.close()
            return '\n'.join(text_lines)
        except ImportError:
            logging.warning("⚠️ PyMuPDF not installed")
            return ""
        except Exception as e:
            logging.error(f"❌ PyMuPDF extraction error: {e}")
            return ""
    
    def _extract_with_pypdf2(self, pdf_path: str) -> str:
        """Extract text using PyPDF2"""
        try:
            text_lines = []
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                for page_num, page in enumerate(pdf_reader.pages):
                    try:
                        page_text = page.extract_text()
                        if page_text:
                            lines = page_text.split('\n')
                            for line in lines:
                                line = line.strip()
                                if line:
                                    text_lines.append(line)
                    except Exception as e:
                        logging.warning(f"⚠️ Failed to extract page {page_num + 1}: {e}")
                        continue
            return '\n'.join(text_lines)
        except Exception as e:
            logging.error(f"❌ PyPDF2 extraction error: {e}")
            return ""

    def create_enhanced_chunks(self, text: str, source: str) -> List[KnowledgeChunk]:
        """Create enhanced knowledge chunks from extracted text"""
        chunks = []
        lines = text.split('\n')
        
        current_section = ""
        section_content = []
        
        for line in lines:
            line = line.strip()
            if not line or len(line) < 5:
                continue
            
            # Detect section headers
            if any(header in line.upper() for header in [
                'EDUCATIONAL DETAILS', 'EDUCATION', 'PERSONAL DETAILS', 'PERSONAL INFO',
                'EXPERIENCE', 'WORK EXPERIENCE', 'PROJECTS', 'SKILLS', 'CONTACT'
            ]):
                # Process previous section
                if current_section and section_content:
                    self._create_section_chunks(current_section, section_content, source, chunks)
                
                current_section = line
                section_content = []
                continue
            
            # Add line to current section
            if current_section:
                section_content.append(line)
            else:
                # Create individual chunks for lines without section
                chunk = self._create_individual_chunk(line, source)
                if chunk:
                    chunks.append(chunk)
        
        # Process final section
        if current_section and section_content:
            self._create_section_chunks(current_section, section_content, source, chunks)
        
        return chunks
    
    def _create_section_chunks(self, section_header: str, content_lines: List[str], source: str, chunks: List[KnowledgeChunk]):
        """Create chunks for a section"""
        section_text = ' '.join(content_lines)
        category = self._categorize_content(section_text)
        
        # Create comprehensive section chunk
        chunk = KnowledgeChunk(
            content=section_text,
            context=f"{section_header} section",
            category=category,
            keywords=self.extract_keywords_dynamically(section_text),
            source=source,
            raw_text=section_text,
            metadata={'section': section_header, 'type': 'section'}
        )
        chunks.append(chunk)
    
    def _create_individual_chunk(self, line: str, source: str, section: str = None) -> Optional[KnowledgeChunk]:
        """Create individual chunk from a line"""
        if len(line.strip()) < 10:
            return None
            
        category = self._categorize_content(line)
        context = f"{section} - {category}" if section else category
        
        return KnowledgeChunk(
            content=line.strip(),
            context=context,
            category=category,
            keywords=self.extract_keywords_dynamically(line),
            source=source,
            raw_text=line.strip(),
            metadata={'section': section, 'type': 'line'}
        )

    def find_relevant_chunks(self, query: str, top_k: int = 8) -> List[Tuple[KnowledgeChunk, float]]:
        """IMPROVED chunk finding with intent-aware scoring and guaranteed results"""
        query_lower = query.lower().strip()
        scored_chunks = []
        
        logging.info(f"🔍 Searching for: '{query}' in {len(self.knowledge_chunks)} chunks")
        
        # Handle greetings
        greetings = ['hi', 'hello', 'hey', 'hola', 'good morning', 'good afternoon', 'good evening', 'namaste']
        if any(greeting in query_lower for greeting in greetings):
            return [(KnowledgeChunk(
                content="Hello! How can I help you learn about Bharavi's portfolio today?",
                context="greeting",
                category="greeting",
                keywords=["greeting"],
                source="system",
                raw_text="Hello! How can I help you learn about Bharavi's portfolio today?"
            ), 10.0)]
        
        # Query preprocessing
        corrected_query, was_corrected, corrections = self.advanced_spell_correction(query)
        query_words = set(re.findall(r'\b\w+\b', corrected_query.lower()))
        
        # ENHANCED KEYWORD MATCHING - More comprehensive patterns with intent awareness
        keyword_patterns = {
            # Personal information patterns - HIGH PRIORITY
            'name': ['name', 'called', 'who'],
            'age': ['age', 'old', 'years old'],
            'birth_date': ['birth', 'born', 'birthday', 'date of birth'],
            
            # Education patterns - HIGH PRIORITY
            'degree': ['degree', 'bachelor', 'btech', 'graduation', 'engineering', 'course', 'study'],
            'college': ['college', 'university', 'institution', 'school'],
            'education': ['education', 'qualification', 'academic', 'educational'],
            '12th': ['12th', 'twelfth', 'intermediate', 'higher secondary'],
            '10th': ['10th', 'tenth', 'secondary', 'matriculation'],
            
            # Skills patterns
            'skills': ['skill', 'skills', 'technology', 'technologies', 'programming', 'technical'],
            'all_skills': ['all skills', 'complete skills', 'total skills', 'skill set'],
            
            # Experience patterns  
            'experience': ['experience', 'work', 'job', 'internship', 'position'],
            'company': ['company', 'companies', 'organization', 'firm'],
            
            # Project patterns
            'projects': ['project', 'projects', 'work', 'built', 'developed'],
            
            # Location patterns
            'location': ['location', 'place', 'where', 'live'],
            'native': ['native', 'from', 'hometown', 'origin'],
            
            # Contact patterns
            'contact': ['contact', 'reach', 'email', 'phone', 'linkedin', 'github']
        }
        
        # Score each chunk with intent-aware scoring
        for chunk in self.knowledge_chunks:
            score = 0.0
            chunk_content_lower = chunk.content.lower()
            
            # 1. EXACT PHRASE MATCHING (Highest priority)
            if corrected_query in chunk_content_lower:
                score += 30.0
                logging.info(f"✅ Exact match found in chunk: {chunk.content[:50]}...")
            
            # 2. INTENT-AWARE KEYWORD PATTERN MATCHING (Enhanced)
            for pattern_name, patterns in keyword_patterns.items():
                pattern_matched = False
                query_has_pattern = any(pattern in corrected_query for pattern in patterns)
                chunk_has_pattern = any(pattern in chunk_content_lower for pattern in patterns)
                
                if query_has_pattern and chunk_has_pattern:
                    # Personal information gets highest priority
                    if pattern_name in ['name', 'age', 'birth_date']:
                        if chunk.metadata and chunk.metadata.get('field') == pattern_name:
                            score += 25.0  # Perfect match for specific personal info
                        else:
                            score += 20.0
                        pattern_matched = True
                    elif pattern_name in ['degree', 'college']:
                        score += 22.0  # High priority for degree/college queries
                        pattern_matched = True
                    elif pattern_name in ['education', '12th', '10th']:
                        score += 18.0
                        pattern_matched = True
                    elif pattern_name == 'all_skills' and chunk.metadata and chunk.metadata.get('category') == 'all':
                        score += 20.0
                        pattern_matched = True
                    elif pattern_name in ['skills', 'experience', 'projects']:
                        score += 15.0
                        pattern_matched = True
                    else:
                        score += 10.0
                        pattern_matched = True
                
                if pattern_matched:
                    logging.info(f"📋 Pattern '{pattern_name}' matched in chunk: {chunk.content[:30]}...")
            
            # 3. METADATA-BASED SCORING (Enhanced for personal info)
            if chunk.metadata:
                metadata_type = chunk.metadata.get('type')
                metadata_field = chunk.metadata.get('field')
                
                # Personal information metadata matching
                if metadata_type == 'basic_info':
                    if 'name' in corrected_query and metadata_field == 'name':
                        score += 20.0
                    elif any(age_term in corrected_query for age_term in ['age', 'old']) and metadata_field == 'age':
                        score += 20.0
                    elif any(birth_term in corrected_query for birth_term in ['birth', 'born', 'birthday']) and metadata_field == 'birth_date':
                        score += 20.0
                
                # Education metadata matching
                if metadata_type == 'education':
                    if any(edu_word in corrected_query for edu_word in ['education', 'degree', 'college']):
                        score += 8.0
                    if chunk.metadata.get('level') == 'degree' and any(deg_word in corrected_query for deg_word in ['degree', 'bachelor', 'btech']):
                        score += 12.0
                    if chunk.metadata.get('level') == 'college' and 'college' in corrected_query:
                        score += 12.0
                    if chunk.metadata.get('level') == '12th' and ('12th' in corrected_query or 'twelfth' in corrected_query):
                        score += 12.0
                    if chunk.metadata.get('level') == '10th' and ('10th' in corrected_query or 'tenth' in corrected_query):
                        score += 12.0
                
                # Experience metadata matching
                if metadata_type == 'experience':
                    if any(exp_word in corrected_query for exp_word in ['experience', 'work', 'job', 'internship']):
                        score += 8.0
                    # Company-specific matching
                    company_key = chunk.metadata.get('company')
                    if company_key:
                        exp_data = self.portfolio_data['experience'].get(company_key, {})
                        if exp_data.get('company', '').lower() in corrected_query:
                            score += 15.0
            
            # 4. WORD OVERLAP SCORING
            chunk_words = set(re.findall(r'\b\w+\b', chunk_content_lower))
            word_overlap = len(query_words.intersection(chunk_words))
            if word_overlap > 0:
                score += word_overlap * 4.0
                logging.info(f"🔤 Word overlap ({word_overlap}) in chunk: {chunk.content[:30]}...")
            
            # 5. KEYWORD MATCHING (from chunk keywords)
            if chunk.keywords:
                chunk_keywords = set(chunk.keywords)
                query_keywords = set(self.extract_keywords_dynamically(corrected_query))
                keyword_overlap = len(chunk_keywords.intersection(query_keywords))
                if keyword_overlap > 0:
                    score += keyword_overlap * 5.0
            
            # 6. CATEGORY RELEVANCE BOOST
            category_boosts = {
                'personal': ['name', 'about', 'who', 'age', 'old', 'birth', 'born'],
                'education': ['education', 'degree', 'qualification', 'study', 'college', '10th', '12th'],
                'skills': ['skill', 'technology', 'programming', 'technical'],
                'experience': ['experience', 'work', 'job', 'internship'],
                'projects': ['project', 'built', 'developed', 'created'],
                'location': ['location', 'place', 'where', 'from'],
                'contact': ['contact', 'email', 'phone', 'reach']
            }
            
            for category, boost_words in category_boosts.items():
                if chunk.category == category and any(word in corrected_query for word in boost_words):
                    score += 8.0
            
            if score > 0:
                scored_chunks.append((chunk, score))
                logging.info(f"📊 Chunk scored {score:.1f}: {chunk.content[:40]}...")
        
        # Add semantic similarity if available
        if self.model and len(self.embeddings) > 0:
            try:
                query_embedding = self.model.encode([corrected_query])
                similarities = cosine_similarity(query_embedding, self.embeddings)[0]
                
                for i, similarity in enumerate(similarities):
                    if similarity > 0.3 and i < len(self.knowledge_chunks):
                        existing_chunk = self.knowledge_chunks[i]
                        
                        # Check if this chunk already has a score
                        found = False
                        for j, (chunk, score) in enumerate(scored_chunks):
                            if chunk.content == existing_chunk.content:
                                scored_chunks[j] = (chunk, score + (similarity * 8.0))
                                found = True
                                break
                        
                        if not found and similarity > 0.4:
                            scored_chunks.append((existing_chunk, similarity * 8.0))
                            
            except Exception as e:
                logging.error(f"Semantic search error: {e}")
        
        # Sort and filter
        scored_chunks.sort(key=lambda x: x[1], reverse=True)
        
        # Log top results
        logging.info(f"🎯 Top scoring chunks for '{query}':")
        for i, (chunk, score) in enumerate(scored_chunks[:5]):
            logging.info(f"   {i+1}. Score: {score:.1f} | {chunk.content[:60]}...")
        
        filtered_chunks = [(chunk, score) for chunk, score in scored_chunks if score > 1.0]
        
        if not filtered_chunks:
            logging.warning(f"⚠️ No chunks found for query: '{query}'. Available chunks: {len(self.knowledge_chunks)}")
            # Return at least the highest scoring chunk if any exist
            if scored_chunks:
                filtered_chunks = [scored_chunks[0]]
        
        logging.info(f"✅ Returning {len(filtered_chunks)} relevant chunks for '{query}'")
        return filtered_chunks[:top_k]

    def generate_comprehensive_response(self, query: str, relevant_chunks: List[Tuple[KnowledgeChunk, float]]) -> str:
        """Generate comprehensive responses with better context handling - FALLBACK METHOD"""
        if not relevant_chunks:
            return "I don't have specific information about that. You can ask about education, experience, projects, skills, or contact details."
    
        query_lower = query.lower()
        
        # Handle specific query types with enhanced responses
        
        # DEGREE/COLLEGE queries - prioritize education chunks
        if any(term in query_lower for term in ['degree', 'college', 'university', 'course', 'study']):
            education_chunks = [(chunk, score) for chunk, score in relevant_chunks if chunk.category == 'education']
            if education_chunks:
                best_edu_chunk, _ = education_chunks[0]
                response = self.remove_personal_pronouns_comprehensive(best_edu_chunk.content)
                return response
        
        # ALL SKILLS query
        if any(pattern in query_lower for pattern in ['all skills', 'complete skills', 'skills list', 'what skills']):
            return self.format_conversational_skills_response()
        
        # EDUCATION query (general)
        if any(edu_term in query_lower for edu_term in ['education', 'qualification', 'academic']):
            return self.format_conversational_education_response(relevant_chunks)
        
        # EXPERIENCE query
        if any(exp_term in query_lower for exp_term in ['experience', 'work', 'internship', 'job']):
            return self.format_conversational_experience_response(relevant_chunks)
        
        # PROJECT query
        if any(proj_term in query_lower for proj_term in ['project', 'projects', 'built', 'developed']):
            return self.format_conversational_project_response(relevant_chunks)
        
        # CONTACT query
        if any(contact_term in query_lower for contact_term in ['contact', 'email', 'phone', 'reach']):
            return self.format_conversational_contact_response()
        
        # LOCATION query
        if any(loc_term in query_lower for loc_term in ['location', 'place', 'where', 'from', 'native']):
            return self.format_conversational_location_response(relevant_chunks, query_lower)
        
        # PERSONAL/NAME query
        if any(personal_term in query_lower for personal_term in ['name', 'who', 'about']):
            return self.format_conversational_personal_response()
        
        # Default: Return best matching chunk with conversational tone
        best_chunk, _ = relevant_chunks[0]
        response = self.remove_personal_pronouns_comprehensive(best_chunk.content)
        response = self.remove_duplicate_content(response)
        return response

    def format_conversational_skills_response(self) -> str:
        """Format comprehensive skills response"""
        skills_data = self.portfolio_data['skills']
        
        response_parts = ["Bharavi has expertise in multiple technical areas:"]
        
        # Programming languages
        if 'programming' in skills_data:
            programming_skills = skills_data['programming']
            response_parts.append(f"Programming languages: {', '.join(programming_skills)}.")
        
        # Data science skills
        if 'data_science' in skills_data:
            ds_skills = skills_data['data_science']  
            response_parts.append(f"Data science specializations: {', '.join(ds_skills)}.")
        
        # Frameworks and tools
        if 'frameworks' in skills_data:
            framework_skills = skills_data['frameworks']
            response_parts.append(f"Frameworks and libraries: {', '.join(framework_skills)}.")
        
        # Databases
        if 'databases' in skills_data:
            db_skills = skills_data['databases']
            response_parts.append(f"Database technologies: {', '.join(db_skills)}.")
        
        # Tools
        if 'tools' in skills_data:
            tool_skills = skills_data['tools']
            response_parts.append(f"Development tools: {', '.join(tool_skills)}.")
        
        # Cloud platforms
        if 'cloud' in skills_data:
            cloud_skills = skills_data['cloud']
            response_parts.append(f"Cloud platforms: {', '.join(cloud_skills)}.")
        
        return " ".join(response_parts)

    def format_conversational_education_response(self, chunks: List[Tuple[KnowledgeChunk, float]]) -> str:
        """Format complete educational background"""
        education_info = {
            'degree': None,
            '12th': None,
            '10th': None
        }
        
        # Extract education information from chunks
        for chunk, score in chunks:
            content_lower = chunk.content.lower()
            
            if any(degree_term in content_lower for degree_term in ['bachelor', 'btech', 'degree']) and not education_info['degree']:
                education_info['degree'] = self.remove_personal_pronouns_comprehensive(chunk.content)
            elif '12th' in content_lower and not education_info['12th']:
                education_info['12th'] = self.remove_personal_pronouns_comprehensive(chunk.content)
            elif '10th' in content_lower and not education_info['10th']:
                education_info['10th'] = self.remove_personal_pronouns_comprehensive(chunk.content)
        
        response_parts = ["Bharavi's educational background:"]
        
        if education_info['degree']:
            response_parts.append(education_info['degree'])
        
        if education_info['12th']:
            response_parts.append(education_info['12th'])
        
        if education_info['10th']:
            response_parts.append(education_info['10th'])
        
        return " ".join(response_parts)

    def format_conversational_experience_response(self, chunks: List[Tuple[KnowledgeChunk, float]]) -> str:
        """Format experience response"""
        experience_data = self.portfolio_data['experience']
        
        response_parts = [f"Bharavi has professional experience at {len(experience_data)} companies:"]
        
        for exp_key, exp_data in experience_data.items():
            company = exp_data['company']
            position = exp_data['position']
            duration = exp_data.get('duration', '')
            description = exp_data['description']
            
            clean_description = self.remove_personal_pronouns_comprehensive(description)
            
            exp_summary = f"At {company}, she worked as a {position}"
            if duration:
                exp_summary += f" for {duration}"
            exp_summary += f". {clean_description}"
            
            response_parts.append(exp_summary)
        
        return " ".join(response_parts)

    def format_conversational_project_response(self, chunks: List[Tuple[KnowledgeChunk, float]]) -> str:
        """Format project response"""
        projects_data = self.portfolio_data['projects']
        
        response_parts = [f"Bharavi has worked on {len(projects_data)} major projects:"]
        
        for proj_key, proj_data in projects_data.items():
            project_name = proj_data['name']
            description = proj_data['description']
            technologies = proj_data.get('technologies', [])
            
            clean_description = self.remove_personal_pronouns_comprehensive(description)
            
            proj_summary = f"{project_name}: {clean_description}"
            if technologies:
                proj_summary += f" Technologies: {', '.join(technologies)}."
            
            response_parts.append(proj_summary)
        
        return " ".join(response_parts)

    def format_conversational_location_response(self, chunks: List[Tuple[KnowledgeChunk, float]], query: str) -> str:
        """Format location response"""
        basic_info = self.portfolio_data['basic_info']
        
        if any(term in query for term in ['native', 'from', 'hometown', 'origin']):
            return f"Bharavi is originally from {basic_info['native']}."
        elif any(term in query for term in ['current', 'live', 'now', 'location']):
            return f"Bharavi currently lives in {basic_info['location']}."
        else:
            return f"Bharavi is from {basic_info['native']} and currently lives in {basic_info['location']}."

    def format_conversational_contact_response(self) -> str:
        """Format contact response"""
        contact_info = self.portfolio_data['contact']
        
        response_parts = ["You can reach Bharavi through:"]
        
        if 'email' in contact_info:
            response_parts.append(f"Email: {contact_info['email']}.")
        
        if 'phone' in contact_info:
            response_parts.append(f"Phone: {contact_info['phone']}.")
        
        if 'linkedin' in contact_info:
            response_parts.append(f"LinkedIn: {contact_info['linkedin']}.")
        
        if 'github' in contact_info:
            response_parts.append(f"GitHub: {contact_info['github']}.")
        
        return " ".join(response_parts)

    def format_conversational_personal_response(self) -> str:
        """Format personal information response"""
        basic_info = self.portfolio_data['basic_info']
        
        response_parts = [f"Her name is {basic_info['name']}."]
        response_parts.append(f"She is an {basic_info['designation']}.")
        
        clean_bio = self.remove_personal_pronouns_comprehensive(basic_info['bio'])
        response_parts.append(clean_bio)
        
        return " ".join(response_parts)

    def query(self, user_query: str, session_id: str = 'default') -> Dict:
        """Main query method with enhanced intent detection and precise responses"""
        try:
            # Validate knowledge base
            if not self.knowledge_chunks:
                logging.error("❌ No knowledge chunks available!")
                return {
                    'answer': "Portfolio content is not available. Please ensure the system is properly configured.",
                    'sources': [],
                    'confidence': False,
                    'spell_corrected': False,
                    'error': True
                }
            
            query = user_query.strip()
            if not query:
                return {
                    'answer': "Please ask something specific about education, skills, experience, projects, or contact information.",
                    'sources': [],
                    'confidence': False,
                    'spell_corrected': False
                }
            
            # Log the query attempt
            logging.info(f"🎯 Processing query: '{query}' (Knowledge chunks: {len(self.knowledge_chunks)})")
            
            # Advanced spell correction
            corrected_query, was_corrected, corrections_made = self.advanced_spell_correction(query)
            
            if was_corrected:
                logging.info(f"📝 Spell corrected: '{query}' → '{corrected_query}'")
            
            # ENHANCED: Detect query intent and entities
            intent_data = self.detect_query_intent(corrected_query)
            logging.info(f"🧠 Detected intent: {intent_data['intent']}, Response type: {intent_data['response_type']}")
            
            # Find relevant chunks
            relevant_chunks = self.find_relevant_chunks(corrected_query, top_k=10)
            
            if not relevant_chunks:
                logging.warning(f"⚠️ No relevant chunks found for: '{corrected_query}'")
                # Provide helpful fallback
                return {
                    'answer': "I couldn't find specific information about that. You can ask about Bharavi's education, skills, work experience, projects, or contact details.",
                    'sources': [],
                    'confidence': False,
                    'spell_corrected': was_corrected,
                    'corrections_made': corrections_made if was_corrected else [],
                    'suggestions': [
                        "What is her name?",
                        "How old is she?",
                        "What is her degree?",
                        "Which college did she attend?",
                        "Tell me about her skills",
                        "What work experience does she have?",
                        "What projects has she worked on?",
                        "How can I contact her?"
                    ]
                }
            
            # ENHANCED: Generate precise, intent-aware response
            answer = self.generate_precise_response(corrected_query, intent_data, relevant_chunks)
            
            # Final cleanup
            answer = self.remove_duplicate_content(answer)
            if not any(pronoun in answer.lower() for pronoun in ['she', 'her', 'bharavi']):
                answer = self.remove_personal_pronouns_comprehensive(answer)
            
            # Calculate confidence
            max_score = max(score for _, score in relevant_chunks)
            confidence = max_score > 3.0
            
            sources = list(set(chunk.source for chunk, _ in relevant_chunks))
            
            # Store conversation context
            if session_id not in self.conversation_history:
                self.conversation_history[session_id] = []
            
            self.conversation_history[session_id].append({
                'query': user_query,
                'corrected_query': corrected_query if was_corrected else None,
                'intent': intent_data['intent'],
                'answer': answer,
                'timestamp': datetime.now().isoformat()
            })
            
            # Log successful response
            logging.info(f"✅ Generated response for '{query}' (intent: {intent_data['intent']}, confidence: {confidence}, score: {max_score:.1f})")
            
            return {
                'answer': answer,
                'sources': sources,
                'confidence': confidence,
                'relevance_score': float(max_score),
                'chunks_used': len(relevant_chunks),
                'spell_corrected': was_corrected,
                'corrections_made': corrections_made if was_corrected else [],
                'original_query': user_query if was_corrected else None,
                'corrected_query': corrected_query if was_corrected else None,
                'intent_detected': intent_data['intent'],
                'response_type': intent_data['response_type'],
                'conversation_context': len(self.conversation_history.get(session_id, [])),
                'error': False
            }
            
        except Exception as e:
            logging.error(f"❌ Query processing error: {e}")
            traceback.print_exc()
            return {
                'answer': "I encountered an issue while processing your question. Please try rephrasing it or ask about education, skills, experience, projects, or contact information.",
                'sources': [],
                'confidence': False,
                'spell_corrected': False,
                'error': True,
                'error_details': str(e)
            }


# Simple fallback chatbot
class SimplePortfolioChatbot:
    def query(self, query: str, session_id: str = 'default') -> Dict:
        return {
            'answer': "Portfolio information is not available. Please ensure PDF files are in the static folder.",
            'sources': [],
            'confidence': False,
            'spell_corrected': False
        }


# Flask app setup
app = Flask(__name__)
app.config['STATIC_FOLDER'] = os.path.join(os.path.dirname(__file__), 'static')
app.secret_key = 'your_secret_key_here'

# Initialize chatbots
simple_chatbot = SimplePortfolioChatbot()
USE_RAG = False
rag_chatbot = None

try:
    openai_key = os.environ.get('OPENAI_API_KEY')
    rag_chatbot = EnhancedPortfolioRAGChatbot(
        static_folder=app.config['STATIC_FOLDER'],
        openai_api_key=openai_key
    )
    
    if hasattr(rag_chatbot, 'knowledge_chunks') and len(rag_chatbot.knowledge_chunks) > 0:
        USE_RAG = True
        print("✅ Enhanced RAG Chatbot with Intent Detection Initialized Successfully!")
        print(f"📚 Loaded {len(rag_chatbot.knowledge_chunks)} knowledge chunks")
        
        # Log chunk categories for debugging
        categories = {}
        for chunk in rag_chatbot.knowledge_chunks:
            categories[chunk.category] = categories.get(chunk.category, 0) + 1
        print("📊 Chunk categories:", categories)
        
    else:
        USE_RAG = False
        print("⚠️ RAG Chatbot initialized but no content loaded!")
except Exception as e:
    print(f"⚠️ RAG initialization failed: {e}")
    traceback.print_exc()

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/api/chat', methods=['POST'])
def chat():
    try:
        data = request.get_json()
        query = data.get('query', '').strip()
        session_id = data.get('session_id', 'default')
        
        if not query:
            return jsonify({'error': 'Please ask something!'}), 400
        
        if USE_RAG and rag_chatbot:
            response = rag_chatbot.query(query, session_id)
            # Ensure all values are JSON serializable
            for key, value in response.items():
                if isinstance(value, np.bool_):
                    response[key] = bool(value)
                elif isinstance(value, (np.float32, np.float64)):
                    response[key] = float(value)
                elif isinstance(value, (np.int32, np.int64)):
                    response[key] = int(value)
            return jsonify(response)
        else:
            answer_dict = simple_chatbot.query(query, session_id)
            return jsonify(answer_dict)
            
    except Exception as e:
        logging.error(f"Chat API error: {e}")
        return jsonify({
            "error": "Something went wrong",
            "details": str(e)
        }), 500

@app.route('/api/debug', methods=['GET'])
def debug_info():
    """Enhanced debug endpoint with detailed chunk information and intent detection"""
    try:
        if USE_RAG and rag_chatbot:
            categories = defaultdict(int)
            category_samples = defaultdict(list)
            
            for chunk in rag_chatbot.knowledge_chunks:
                categories[chunk.category] += 1
                if len(category_samples[chunk.category]) < 3:
                    category_samples[chunk.category].append({
                        'content': chunk.content[:100] + '...' if len(chunk.content) > 100 else chunk.content,
                        'keywords': chunk.keywords[:5] if chunk.keywords else [],
                        'source': chunk.source,
                        'metadata': chunk.metadata
                    })
            
            debug_data = {
                'status': 'Enhanced RAG Chatbot with Intent Detection & Precise Responses',
                'total_chunks': len(rag_chatbot.knowledge_chunks),
                'categories': dict(categories),
                'category_samples': dict(category_samples),
                'embedding_shape': str(rag_chatbot.embeddings.shape) if hasattr(rag_chatbot, 'embeddings') and hasattr(rag_chatbot.embeddings, 'shape') else 'Not available',
                'static_folder': app.config['STATIC_FOLDER'],
                'static_folder_exists': os.path.exists(app.config['STATIC_FOLDER']),
                'portfolio_data_loaded': bool(rag_chatbot.portfolio_data),
                'education_data_available': bool(rag_chatbot.portfolio_data.get('education')),
                'personal_info_enhanced': bool(rag_chatbot.portfolio_data.get('basic_info', {}).get('age')),
                'raw_content_files': list(rag_chatbot.raw_content.keys()) if hasattr(rag_chatbot, 'raw_content') else [],
                'spell_corrections_available': len(rag_chatbot.spell_corrections),
                'conversation_sessions': len(rag_chatbot.conversation_history),
                'new_features_implemented': [
                    '✅ INTENT DETECTION - Understands specific user queries',
                    '✅ PRECISE RESPONSES - Gives exactly what user asks for',
                    '✅ ENHANCED PERSONAL INFO - Age, birth date, detailed education',
                    '✅ CONTEXT-AWARE SCORING - Better chunk matching',
                    '✅ SCHOOL NAMES INCLUDED - Complete education details',
                    '✅ COMPANY-SPECIFIC QUERIES - Detailed experience info',
                    '✅ IMPROVED SPELL CORRECTION - Better query understanding',
                    '✅ METADATA-BASED MATCHING - More accurate responses',
                    '✅ CONVERSATIONAL CONTINUITY - Better context handling',
                    '✅ LOGICAL QUESTION SUPPORT - Age calculations, etc.'
                ],
                'supported_intents': [
                    'name - Gets exact name',
                    'age - Calculates and returns current age', 
                    'birth_date - Returns formatted birth date',
                    'degree - Specific degree information',
                    'college - College/university details',
                    '12th/10th - Grade-specific info with school names',
                    'experience - Work experience (general or company-specific)',
                    'skills - Technical skills (comprehensive or summary)',
                    'projects - Project details',
                    'contact - Contact information (specific or all)',
                    'location - Current location or native place'
                ],
                'test_queries_enhanced': [
                    'What is her name?',
                    'How old is she?',
                    'When was she born?',
                    'What is her degree?',
                    'Which college did she attend?',
                    'What is her 12th percentage?',
                    'Which school did she go to for 12th?',
                    'Tell me about her experience at ICSR',
                    'What technologies does she know?',
                    'How can I contact her?',
                    'Where is she from?',
                    'What projects has she built?'
                ],
                'chunk_distribution': {
                    'personal_chunks': len([c for c in rag_chatbot.knowledge_chunks if c.category == 'personal']),
                    'education_chunks': len([c for c in rag_chatbot.knowledge_chunks if c.category == 'education']),
                    'skills_chunks': len([c for c in rag_chatbot.knowledge_chunks if c.category == 'skills']),
                    'experience_chunks': len([c for c in rag_chatbot.knowledge_chunks if c.category == 'experience']),
                    'project_chunks': len([c for c in rag_chatbot.knowledge_chunks if c.category == 'projects']),
                    'contact_chunks': len([c for c in rag_chatbot.knowledge_chunks if c.category == 'contact']),
                    'location_chunks': len([c for c in rag_chatbot.knowledge_chunks if c.category == 'location'])
                }
            }
            
            return jsonify(debug_data)
        
        return jsonify({
            'error': 'RAG not available',
            'static_folder': app.config['STATIC_FOLDER'],
            'static_folder_exists': os.path.exists(app.config['STATIC_FOLDER'])
        }), 404
    except Exception as e:
        logging.error(f"Debug endpoint error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'healthy',
        'rag_enabled': bool(USE_RAG),
        'version': 'Enhanced Portfolio Assistant v5.0 - INTENT DETECTION & PRECISE RESPONSES',
        'major_enhancements': [
            '🧠 INTENT DETECTION - Understands what user specifically wants',
            '🎯 PRECISE RESPONSES - No more extra information',
            '👤 ENHANCED PERSONAL INFO - Age, birth date, detailed background',
            '🏫 COMPLETE EDUCATION - School names, college details, percentages',
            '🏢 DETAILED EXPERIENCE - Company-specific information',
            '📞 SMART CONTACT - Specific contact method or comprehensive',
            '📍 LOCATION AWARENESS - Native vs current location',
            '🔍 CONTEXT-AWARE SEARCH - Better chunk matching',
            '💬 CONVERSATIONAL FLOW - Natural dialogue support',
            '🧮 LOGICAL CALCULATIONS - Age from birth date, etc.'
        ],
        'key_improvements': {
            'intent_detection': 'Automatically detects what user wants (name, age, degree, etc.)',
            'precise_responses': 'Gives exactly what is asked, no extra info',
            'enhanced_data': 'Added age calculation, birth date, school names',
            'context_awareness': 'Better understanding of specific vs general queries',
            'conversation_flow': 'Maintains context across conversation'
        },
        'message': 'Now supports precise, context-aware responses with intent detection!',
        'total_chunks': len(rag_chatbot.knowledge_chunks) if rag_chatbot else 0,
        'chunk_status': 'LOADED WITH ENHANCED PERSONAL INFO' if rag_chatbot and len(rag_chatbot.knowledge_chunks) > 0 else 'NOT LOADED'
    })

if __name__ == '__main__':
    os.makedirs(app.config['STATIC_FOLDER'], exist_ok=True)
    os.makedirs('templates', exist_ok=True)
    
    print("🚀 Starting Enhanced Portfolio Assistant v5.0 - INTENT DETECTION & PRECISE RESPONSES...")
    print("📁 Static folder:", app.config['STATIC_FOLDER'])
    
    if USE_RAG:
        print("\n✅ MAJOR ENHANCEMENTS IMPLEMENTED:")
        print("   🧠 INTENT DETECTION - Understands specific user queries")
        print("   🎯 PRECISE RESPONSES - Gives exactly what user asks for")
        print("   👤 ENHANCED PERSONAL INFO - Age, birth date, detailed info")
        print("   🏫 COMPLETE EDUCATION - School names included")
        print("   🏢 DETAILED EXPERIENCE - Company-specific queries supported")
        print("   📞 SMART CONTACT - Specific contact methods")
        print("   📍 LOCATION AWARENESS - Native vs current place")
        print("   🔍 CONTEXT-AWARE SEARCH - Better chunk matching")
        
        print(f"\n📊 ENHANCED KNOWLEDGE BASE:")
        print(f"   📚 Total chunks: {len(rag_chatbot.knowledge_chunks)}")
        print(f"   🎂 Age calculation: {rag_chatbot.portfolio_data['basic_info']['age']} years old")
        print(f"   📅 Birth date: {rag_chatbot.portfolio_data['basic_info']['date_of_birth']}")
        
        if rag_chatbot.knowledge_chunks:
            categories = {}
            for chunk in rag_chatbot.knowledge_chunks:
                categories[chunk.category] = categories.get(chunk.category, 0) + 1
            for category, count in categories.items():
                print(f"   📋 {category}: {count} chunks")
        
        print("\n🧪 TEST THESE ENHANCED QUERIES:")
        print("   • 'What is her name?' - Returns: Her name is Bharavi")
        print("   • 'How old is she?' - Returns: She is X years old")
        print("   • 'When was she born?' - Returns: She was born on [date]")
        print("   • 'What is her degree?' - Returns degree name only")
        print("   • 'Which college?' - Returns college name and location")
        print("   • 'What is 12th percentage?' - Returns percentage with school")
        print("   • 'Tell me about ICSR experience' - Company-specific details")
        print("   • 'What is her email?' - Returns email only")
        
    else:
        print("❌ RAG Chatbot failed to initialize!")
    
    print("\n🌐 Access URLs:")
    print("   Main App: http://localhost:5000")
    print("   Debug Info: http://localhost:5000/api/debug")
    print("   Health Check: http://localhost:5000/health")
    
    app.run(debug=True, host='0.0.0.0', port=5000)